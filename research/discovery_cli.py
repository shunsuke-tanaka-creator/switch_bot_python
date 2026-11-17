"""Train the shared discovery model from a complete private lifelog export.

Does not deploy a server or overwrite a legacy checkpoint. Use a dedicated
output directory; explicitly identify synthetic input with --data-mode sample.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--export', required=True, help='Complete lifelog-all-v1 JSON export')
    parser.add_argument('--output', required=True, help='Dedicated discovery checkpoint directory')
    parser.add_argument('--at', help='Timezone-aware as-of cutoff; defaults to now')
    parser.add_argument('--epochs', type=int, default=12)
    parser.add_argument('--hidden', type=int, default=48)
    parser.add_argument('--device', choices=('cpu','cuda','mps','auto'), default='cpu')
    parser.add_argument('--data-mode', choices=('real','sample'), default='real')
    parser.add_argument('--text-model', help='Existing local sentence model directory')
    parser.add_argument('--evaluate-test', action='store_true')
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args(argv)
    if args.epochs < 1 or args.hidden < 1:
        parser.error('epochs and hidden must be positive')
    try:
        payload = json.loads(Path(args.export).read_text(encoding='utf-8'))
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    if not isinstance(payload, dict) or payload.get('format')!='lifelog-all-v1' or not isinstance(payload.get('records'),list):
        parser.error('Expected a complete lifelog-all-v1 export')
    participant = payload.get('participant')
    if not participant or any(not isinstance(r,dict) or r.get('participant')!=participant for r in payload['records']):
        parser.error('Export must contain exactly one participant')
    if payload.get('next_cursor') is not None:
        parser.error('Export is paginated/incomplete; collect all pages before training')
    output = Path(args.output)
    if output.is_file():
        parser.error('output must be a dedicated directory')
    existing = output/'status.json'
    checkpoints = ('active.pt','candidate.pt','previous.pt')
    if any((output/name).exists() for name in checkpoints) and not existing.exists():
        parser.error('Refusing existing checkpoints without discovery provenance')
    if existing.exists():
        try:
            previous = json.loads(existing.read_text(encoding='utf-8'))
        except (OSError, ValueError) as exc:
            parser.error(f'Cannot verify output directory: {exc}')
        if not isinstance(previous,dict) or previous.get('architecture')!='discovery':
            parser.error('Refusing to reuse a non-discovery model directory')
    from .discovery_learning import train_discovery
    result = train_discovery(payload['records'],output,args.at or datetime.now(timezone.utc),
                             epochs=args.epochs,hidden=args.hidden,device=args.device,text_model=args.text_model,
                             evaluate_test=args.evaluate_test,force=args.force,
                             data_mode=args.data_mode,contains_sample=args.data_mode=='sample')
    print(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False))


if __name__=='__main__':
    main()
