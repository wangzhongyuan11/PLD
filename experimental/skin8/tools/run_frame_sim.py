"""Compatibility entry point; implementation is in run_sim.py."""
import sys
from run_sim import main
if __name__ == "__main__":
    if len(sys.argv)==1:sys.argv.append("frame_engine")
    main()
