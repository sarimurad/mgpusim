// Command twotenant runs two independent matrix-multiplication workloads on a
// single simulated GPU, so that both tenants share one last-level TLB.
//
// Each call to NewBenchmark calls driver.Init(), which creates a separate
// context and therefore a separate process identifier. Both benchmarks are
// pinned to GPU 1, so any eviction in which the victim's PID differs from the
// filling request's PID is one tenant displacing the other's translation from
// the shared structure.
//
// Usage:
//
//	# control: one tenant only
//	./twotenant -timing -single -x=2048 -y=1024 -z=1024
//
//	# experiment: two co-resident tenants
//	./twotenant -timing -x=2048 -y=1024 -z=1024
//
// The cross-PID eviction count is reported at exit by the L2 TLB
// instrumentation in samples/runner/r9nanobuilder.go.
package main

import (
	"flag"

	_ "net/http/pprof"

	"github.com/sarchlab/mgpusim/v3/benchmarks/amdappsdk/matrixmultiplication"
	"github.com/sarchlab/mgpusim/v3/samples/runner"
)

var xFlag = flag.Uint("x", 2048, "Height of the first matrix, per tenant.")
var yFlag = flag.Uint("y", 1024, "Width of the first and height of the second matrix, per tenant.")
var zFlag = flag.Uint("z", 1024, "Width of the second matrix, per tenant.")
var singleFlag = flag.Bool("single", false,
	"Run only one tenant. Control condition: cross-PID evictions must be zero.")

func main() {
	flag.Parse()

	r := new(runner.Runner).ParseFlag().Init()

	// Tenant A.
	a := matrixmultiplication.NewBenchmark(r.Driver())
	a.X = uint32(*xFlag)
	a.Y = uint32(*yFlag)
	a.Z = uint32(*zFlag)
	a.SelectGPU([]int{1})
	r.AddBenchmarkWithoutSettingGPUsToUse(a)

	// Tenant B, on the same physical GPU and therefore the same L2 TLB.
	if !*singleFlag {
		b := matrixmultiplication.NewBenchmark(r.Driver())
		b.X = uint32(*xFlag)
		b.Y = uint32(*yFlag)
		b.Z = uint32(*zFlag)
		b.SelectGPU([]int{1})
		r.AddBenchmarkWithoutSettingGPUsToUse(b)
	}

	r.Run()
}
