// Command primeprobe models a cross-tenant Prime+Probe attacker against the
// shared last-level TLB and records the resulting [R, 256] contention matrix.
//
// A victim matrix-multiplication workload executes on GPU 1. An attacker,
// modelled at the translation level, primes an eviction set for every set of
// the shared L2 TLB, lets the victim run for one measurement window, then
// probes each set to recover a contention level. Repeating for R windows
// yields the contention matrix that the detection pipeline consumes.
//
// The attacker reads residency directly rather than timing a page walk. The
// two are equivalent -- the latency an attacker times is caused by the
// residency state -- and reading residency avoids perturbing the LRU order,
// since the probe never marks an entry as used.
//
// Usage:
//
//	./primeprobe -timing -rounds=20 -x=1024 -y=1024 -z=1024 -out=matrix.csv
//	./primeprobe -timing -rounds=20 -idle           # control: no victim
package main

import (
	"encoding/csv"
	"flag"
	"fmt"
	"os"
	"strconv"

	_ "net/http/pprof"

	"github.com/tebeka/atexit"
	"github.com/sarchlab/akita/v3/mem/vm"
	"github.com/sarchlab/akita/v3/mem/vm/tlb"
	"github.com/sarchlab/mgpusim/v3/benchmarks/amdappsdk/matrixmultiplication"
	"github.com/sarchlab/mgpusim/v3/samples/runner"
)

var (
	roundsFlag = flag.Int("rounds", 20, "Number of measurement windows R.")
	xFlag      = flag.Uint("x", 1024, "Victim: height of the first matrix.")
	yFlag      = flag.Uint("y", 1024, "Victim: shared dimension.")
	zFlag      = flag.Uint("z", 1024, "Victim: width of the second matrix.")
	idleFlag   = flag.Bool("idle", false, "Run no victim. Control: contention must be ~0.")
	outFlag    = flag.String("out", "matrix.csv", "Output CSV for the [R,256] matrix.")
	attackPID  = flag.Uint("attacker-pid", 9000, "PID the attacker primes under.")
	regionMB   = flag.Uint64("region-mb", 1, "Region stride in MB for eviction sets.")
)

func main() {
	flag.Parse()

	r := new(runner.Runner).ParseFlag().Init()

	l2, ok := r.Platform().GPUs[0].L2TLBs[0].(*tlb.TLB)
	if !ok {
		panic("L2TLB is not a *tlb.TLB; cannot attach the probe")
	}

	numSets := l2.NumSets()
	ways := l2.NumWays()
	stride := *regionMB << 20

	esets := make([]tlb.EvictionSet, numSets)
	for s := 0; s < numSets; s++ {
		esets[s] = l2.BuildEvictionSet(vm.PID(*attackPID), s, ways, stride)
	}

	matrix := make([][]int, *roundsFlag)
	for i := range matrix {
		matrix[i] = make([]int, numSets)
	}

	if *idleFlag {
		for s := range esets {
			l2.Prime(esets[s])
		}
		for round := 0; round < *roundsFlag; round++ {
			for s := range esets {
				matrix[round][s] = l2.Contention(esets[s])
			}
		}
		writeMatrix(*outFlag, matrix)
		fmt.Printf(">>> idle control written to %s (expect all zero)\n", *outFlag)
		return
	}

	victim := matrixmultiplication.NewBenchmark(r.Driver())
	victim.X = uint32(*xFlag)
	victim.Y = uint32(*yFlag)
	victim.Z = uint32(*zFlag)
	victim.SelectGPU([]int{1})
	r.AddBenchmarkWithoutSettingGPUsToUse(victim)

	for s := range esets {
		l2.Prime(esets[s])
	}

	// r.Run() ends with atexit.Exit(0) and never returns, so the probe is
	// registered as an atexit handler. It fires after the victim has finished
	// but before the process exits.
	atexit.Register(func() {
		for s := range esets {
			matrix[0][s] = l2.Contention(esets[s])
		}
		fmt.Printf(">>> after victim: L2 evictValid=%d crossPID=%d subFill=%d\n",
			l2.EvictValid, l2.EvictCrossPID, l2.SubEntryFill)
		writeMatrix(*outFlag, matrix)
		fmt.Printf(">>> contention matrix [%d x %d] written to %s\n",
			*roundsFlag, numSets, *outFlag)
		summarize(matrix, ways)
	})

	r.Run()
}	

		
		
func writeMatrix(path string, matrix [][]int) {
	f, err := os.Create(path)
	if err != nil {
		panic(err)
	}
	defer f.Close()

	w := csv.NewWriter(f)
	defer w.Flush()

	for _, row := range matrix {
		rec := make([]string, len(row))
		for i, v := range row {
			rec[i] = strconv.Itoa(v)
		}
		_ = w.Write(rec)
	}
}

func summarize(matrix [][]int, ways int) {
	total, nonzero, max := 0, 0, 0
	for _, row := range matrix {
		for _, v := range row {
			total += v
			if v > 0 {
				nonzero++
			}
			if v > max {
				max = v
			}
		}
	}
	cells := len(matrix) * len(matrix[0])
	fmt.Printf(">>> contention: mean=%.3f nonzero-cells=%d/%d max=%d (of %d ways)\n",
		float64(total)/float64(cells), nonzero, cells, max, ways)
}
