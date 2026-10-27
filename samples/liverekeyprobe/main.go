// Command liverekeyprobe measures the attacker-side Prime+Probe view of live
// TLB key rotation. Victim fills remain available as a separate counter view.
//
// Rounds are driven from the L2 TLB's RoundHook, so they only advance while
// the TLB is ticking and every run is deterministic. Work done after the last
// full round boundary is written as one final partial row, so the matrix
// always accounts for every victim fill.
//
// Usage:
//
//	./liverekeyprobe -timing -rekey-every=4 -out=runs/k4
//	./liverekeyprobe -timing -idle -out=runs/idle
package main

import (
	"encoding/csv"
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"sort"
	"strconv"

	_ "net/http/pprof"

	"github.com/sarchlab/akita/v3/mem/vm"
	"github.com/sarchlab/akita/v3/mem/vm/tlb"
	"github.com/sarchlab/akita/v3/sim"
	"github.com/sarchlab/mgpusim/v3/benchmarks/amdappsdk/matrixmultiplication"
	"github.com/sarchlab/mgpusim/v3/samples/runner"
	"github.com/tebeka/atexit"
)

var (
	xFlag       = flag.Uint("x", 1024, "Victim matrix height.")
	yFlag       = flag.Uint("y", 1024, "Victim matrix shared dimension.")
	zFlag       = flag.Uint("z", 1024, "Victim matrix width.")
	idleFlag    = flag.Bool("idle", false, "Run without a victim (control).")
	outFlag     = flag.String("out", "liverekeyprobe", "Output file prefix.")
	rekeyEvery  = flag.Int("rekey-every", 0, "Rotate the victim mapping every k rounds; 0 disables rotation.")
	roundCycles = flag.Int("round-cycles", 177984, "Simulated GPU cycles per measurement round.")
	maxRounds   = flag.Int("max-rounds", 0, "Stop recording after N full rounds; 0 records until simulation ends.")
	seedFlag    = flag.Uint64("seed", 1234, "Seed for deterministic key generation.")
	attackerPID = flag.Uint("attacker-pid", 9000, "PID used for the simulated attacker probe.")
	regionMB    = flag.Uint64("region-mb", 1, "Address stride in MB for attacker eviction sets.")
	repeatFlag  = flag.Int("repeat", 1, "Launch the victim kernel this many times on the same buffers, so it stays active for more rounds (k>=16 needs this).")
)

// gpuFreqHz is the R9 Nano GPU clock (r9nanobuilder.go).
const gpuFreqHz = 1e9

type roundRow struct {
	sets     []uint64
	probe    []uint64
	crossPID uint64
	walks    uint64
	flushed  uint64
	epoch    uint64
	full     bool
}

type summary struct {
	Rounds          int     `json:"rounds"`
	FullRounds      int     `json:"full_rounds"`
	ActiveRounds    int     `json:"active_rounds"`
	RekeyEvery      int     `json:"rekey_every"`
	Rotations       int     `json:"rotations"`
	RawTotal        uint64  `json:"raw_total"`
	ProbeContention uint64  `json:"probe_contention_total"`
	VictimFillTotal uint64  `json:"victim_fill_total"`
	FlushedEntries  uint64  `json:"flushed_entries"`
	VictimFills     uint64  `json:"victim_fills"`
	VictimWalks     uint64  `json:"victim_walks"`
	TotalWalks      uint64  `json:"total_walks"`
	SimTimeS        float64 `json:"sim_time_s"`
	VictimPID       uint64  `json:"victim_pid"`
	Idle            bool    `json:"idle"`
	Seed            uint64  `json:"seed"`
	RoundCycles     int     `json:"round_cycles"`
	Repeat          int     `json:"repeat"`
	Capped          bool    `json:"capped_by_max_rounds"`
	AccountingMatch bool    `json:"accounting_match"`
	Measurement     string  `json:"measurement"`
	EvictValid      uint64  `json:"L2_evictValid"`
	CrossPID        uint64  `json:"L2_crossPID"`
	AttackerPID     uint64  `json:"attacker_pid"`
	VictimX         uint32  `json:"victim_x"`
	VictimY         uint32  `json:"victim_y"`
	VictimZ         uint32  `json:"victim_z"`
	SubEntryFill    uint64  `json:"L2_subFill"`
}

// recorder holds the round state shared by the hook and the exit handler.
type recorder struct {
	l2     *tlb.TLB
	period float64

	t0           float64
	started      bool
	fullRounds   int
	activeRounds int
	rotations    int
	keyEpoch     uint64
	victimPID    vm.PID
	haveVictim   bool
	prevWalks    uint64
	prevFills    []uint64
	prevCrossPID uint64
	attackerPID  vm.PID
	esets        []tlb.EvictionSet
	rows         []roundRow
	capped       bool
}

func main() {
	flag.Parse()
	if *roundCycles <= 0 || *rekeyEvery < 0 || *maxRounds < 0 || *regionMB == 0 || *repeatFlag < 1 {
		panic("round-cycles, region-mb and repeat must be positive; rekey-every and max-rounds cannot be negative")
	}

	r := new(runner.Runner).ParseFlag().Init()
	l2, ok := r.Platform().GPUs[0].L2TLBs[0].(*tlb.TLB)
	if !ok {
		panic("L2TLB is not a *tlb.TLB")
	}
	l2.Rekey(0)
	attacker := vm.PID(*attackerPID)
	esets := l2.BuildAllEvictionSets(attacker, l2.NumWays(), *regionMB<<20, 1<<22)
	for s := range esets {
		l2.Prime(esets[s])
	}

	if !*idleFlag {
		victim := matrixmultiplication.NewBenchmark(r.Driver())
		victim.X, victim.Y, victim.Z = uint32(*xFlag), uint32(*yFlag), uint32(*zFlag)
		victim.Repeat = *repeatFlag
		victim.SelectGPU([]int{1})
		r.AddBenchmarkWithoutSettingGPUsToUse(victim)
	}

	rec := &recorder{
		l2: l2, period: float64(*roundCycles) / gpuFreqHz,
		attackerPID: attacker, esets: esets,
	}
	l2.RoundHook = rec.hook

	// r.Run() ends with atexit.Exit(0) and never returns. The idle control
	// goes through the same path, so its rows come from a real run.
	atexit.Register(func() {
		rec.finish()
		writeOutputs(*outFlag, rec, float64(r.Platform().Engine.CurrentTime()))
	})
	r.Run()
}

// hook processes every round boundary that now has passed, in order.
func (rec *recorder) hook(now sim.VTimeInSec) {
	if !rec.started {
		rec.t0, rec.started = float64(now), true
	}
	for float64(now)-rec.t0 >= float64(rec.fullRounds+1)*rec.period {
		if *maxRounds > 0 && rec.fullRounds >= *maxRounds {
			rec.capped = true
			return
		}
		rec.closeRow(true)
	}
}

// finish writes the partial round after the last boundary, unless recording
// was capped by -max-rounds.
func (rec *recorder) finish() {
	if rec.capped {
		return
	}
	rec.findVictim()
	if rec.hasPendingWork() || len(rec.rows) == 0 {
		rec.closeRow(false)
	}
}

// findVictim takes the victim's PID from the walk counters. The victim is the
// only tenant in this run, so the first PID to walk is the victim.
func (rec *recorder) findVictim() {
	if rec.haveVictim || len(rec.l2.WalksByPID) == 0 {
		return
	}
	pids := make([]int, 0, len(rec.l2.WalksByPID))
	for pid := range rec.l2.WalksByPID {
		pids = append(pids, int(pid))
	}
	sort.Ints(pids)
	rec.victimPID, rec.haveVictim = vm.PID(pids[0]), true
	if rec.victimPID == rec.attackerPID {
		panic("attacker PID collides with victim PID")
	}
}

func (rec *recorder) hasPendingWork() bool {
	if !rec.haveVictim {
		return false
	}
	return rec.l2.WalksByPID[rec.victimPID] != rec.prevWalks ||
		sum(rec.l2.FillsByPIDSet[rec.victimPID]) != sum(rec.prevFills)
}

// closeRow records the counters since the previous row, then (for a full
// round) applies a re-key if one is due.
func (rec *recorder) closeRow(full bool) {
	rec.findVictim()

	row := roundRow{
		sets: make([]uint64, rec.l2.NumSets()), probe: make([]uint64, rec.l2.NumSets()),
		epoch: rec.keyEpoch, full: full,
	}
	for s := range rec.esets {
		row.probe[s] = uint64(rec.l2.Contention(rec.esets[s]))
	}
	crossPID := rec.l2.EvictCrossPID
	if crossPID >= rec.prevCrossPID {
		row.crossPID = crossPID - rec.prevCrossPID
	}
	rec.prevCrossPID = crossPID
	if rec.haveVictim {
		fills := rec.l2.FillsByPIDSet[rec.victimPID]
		for s := range row.sets {
			if s < len(fills) {
				prev := uint64(0)
				if s < len(rec.prevFills) {
					prev = rec.prevFills[s]
				}
				row.sets[s] = fills[s] - prev
			}
		}
		walks := rec.l2.WalksByPID[rec.victimPID]
		row.walks = walks - rec.prevWalks
		rec.prevFills = append(rec.prevFills[:0], fills...)
		rec.prevWalks = walks
	}

	if full {
		rec.fullRounds++
		// Count only rounds with measured victim activity. TLB ticks can
		// continue after the benchmark's last translation request; those idle
		// tail rounds must not trigger empty rotations.
		if rec.haveVictim && sum(row.sets) > 0 {
			rec.activeRounds++
			if *rekeyEvery > 0 && rec.activeRounds%*rekeyEvery == 0 {
				rec.keyEpoch++
				rec.l2.RekeyPID(rec.victimPID, splitmix(*seedFlag, rec.keyEpoch))
				row.flushed = uint64(rec.l2.FlushPID(rec.victimPID))
				rec.rotations++
			}
		}
	}

	rec.rows = append(rec.rows, row)
	// Re-prime after recording and after any victim re-key/flush so the next
	// round begins with a full attacker footprint under the unchanged key 0.
	for s := range rec.esets {
		rec.l2.Prime(rec.esets[s])
	}
}

// splitmix derives the key for epoch e from the seed. It never returns 0,
// since key 0 is the published mapping.
func splitmix(seed, epoch uint64) uint64 {
	z := seed + epoch*0x9E3779B97F4A7C15
	z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9
	z = (z ^ (z >> 27)) * 0x94D049BB133111EB
	z ^= z >> 31
	if z == 0 {
		return 1
	}
	return z
}

func sum(xs []uint64) uint64 {
	var t uint64
	for _, x := range xs {
		t += x
	}
	return t
}

func writeOutputs(prefix string, rec *recorder, simTime float64) {
	l2 := rec.l2

	matrixFile, err := os.Create(prefix + "_matrix.csv")
	check(err)
	defer matrixFile.Close()
	mw := csv.NewWriter(matrixFile)
	var raw uint64
	for _, row := range rec.rows {
		record := make([]string, len(row.probe))
		for i, n := range row.probe {
			record[i] = strconv.FormatUint(n, 10)
			raw += n
		}
		check(mw.Write(record))
	}
	mw.Flush()
	check(mw.Error())

	fillFile, err := os.Create(prefix + "_victim_fills.csv")
	check(err)
	defer fillFile.Close()
	fw := csv.NewWriter(fillFile)
	var victimFillTotal uint64
	for _, row := range rec.rows {
		record := make([]string, len(row.sets))
		for i, n := range row.sets {
			record[i] = strconv.FormatUint(n, 10)
			victimFillTotal += n
		}
		check(fw.Write(record))
	}
	fw.Flush()
	check(fw.Error())

	epochFile, err := os.Create(prefix + "_epochs.csv")
	check(err)
	defer epochFile.Close()
	ew := csv.NewWriter(epochFile)
	check(ew.Write([]string{"round", "full_round", "key_epoch", "flushed_this_round",
		"victim_fills_this_round", "victim_walks_this_round", "probe_contention_this_round",
		"cross_pid_evictions_this_round"}))
	for i, row := range rec.rows {
		full := "0"
		if row.full {
			full = "1"
		}
		check(ew.Write([]string{strconv.Itoa(i + 1), full,
			strconv.FormatUint(row.epoch, 10), strconv.FormatUint(row.flushed, 10),
			strconv.FormatUint(sum(row.sets), 10), strconv.FormatUint(row.walks, 10),
			strconv.FormatUint(sum(row.probe), 10), strconv.FormatUint(row.crossPID, 10)}))
	}
	ew.Flush()
	check(ew.Error())

	var victimWalks uint64
	if rec.haveVictim {
		victimWalks = l2.WalksByPID[rec.victimPID]
	}
	s := summary{
		Rounds:          len(rec.rows),
		FullRounds:      rec.fullRounds,
		ActiveRounds:    rec.activeRounds,
		RekeyEvery:      *rekeyEvery,
		Rotations:       rec.rotations,
		RawTotal:        raw,
		ProbeContention: raw,
		VictimFillTotal: victimFillTotal,
		FlushedEntries:  l2.FlushedEntries,
		VictimFills:     victimFillTotal,
		VictimWalks:     victimWalks,
		TotalWalks:      l2.Walks,
		SimTimeS:        simTime,
		VictimPID:       uint64(rec.victimPID),
		Idle:            *idleFlag,
		Seed:            *seedFlag,
		RoundCycles:     *roundCycles,
		Repeat:          *repeatFlag,
		Capped:          rec.capped,
		// Every victim fill must land in exactly one separate accounting row unless capped.
		AccountingMatch: rec.capped || victimFillTotal == sum(l2.FillsByPIDSet[rec.victimPID]),
		Measurement:     "attacker-side prime/probe contention; victim fills also retained separately",
		EvictValid:      l2.EvictValid,
		CrossPID:        l2.EvictCrossPID,
		AttackerPID:     uint64(rec.attackerPID),
		VictimX:         uint32(*xFlag),
		VictimY:         uint32(*yFlag),
		VictimZ:         uint32(*zFlag),
		SubEntryFill:    l2.SubEntryFill,
	}
	b, err := json.MarshalIndent(s, "", "  ")
	check(err)
	check(os.WriteFile(prefix+"_summary.json", append(b, '\n'), 0644))
	fmt.Printf(">>> rounds=%d active=%d rotations=%d raw_total=%d victim_fills=%d "+
		"victim_walks=%d flushed=%d accounting_match=%v sim_time=%.6fs\n",
		s.Rounds, s.ActiveRounds, s.Rotations, s.RawTotal, s.VictimFillTotal,
		s.VictimWalks, s.FlushedEntries, s.AccountingMatch, s.SimTimeS)
	fmt.Printf(">>> L2 evictValid=%d crossPID=%d subFill=%d\n",
		l2.EvictValid, l2.EvictCrossPID, l2.SubEntryFill)
}

func check(err error) {
	if err != nil {
		panic(err)
	}
}
