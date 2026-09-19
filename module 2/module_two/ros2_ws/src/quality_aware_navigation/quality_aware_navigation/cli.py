"""Convert archived module-one CSV to a quality-cost artifact without ROS."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import numpy as np
from .grid import load_module_one
from .model import CostModel, CostParameters


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('csv')
    parser.add_argument('output')
    parser.add_argument('--metadata')
    parser.add_argument('--as-of',type=float,help='Source-clock seconds, default latest observed cell time')
    parser.add_argument('--mode',choices=CostModel.MODES,default='full')
    args=parser.parse_args(argv)
    grid=load_module_one(args.csv,args.metadata)
    as_of=args.as_of if args.as_of is not None else float(grid.last_observed[grid.known].max(initial=0))
    model=CostModel(mode=args.mode)
    costs=model.costs(grid,as_of)
    output=Path(args.output)
    output.mkdir(parents=True,exist_ok=True)
    np.savetxt(output/'costs.csv',costs,fmt='%d',delimiter=',')
    report={'artifact_type':'localization_quality_soft_cost','source':grid.source,
            'spec':asdict(grid.spec),'as_of_source_seconds':as_of,'mode':args.mode,
            'parameters':asdict(model.parameters),'known_cells':int(grid.known.sum()),
            'unknown_cells':int((~grid.known).sum()),'cost_min':int(costs.min()),
            'cost_max':int(costs.max()),'obstacle_map_included':False}
    (output/'metadata.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False,indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
