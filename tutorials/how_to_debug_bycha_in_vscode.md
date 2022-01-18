# How to debug ParaGen in VSCODE

## STEP 1

`pip install debugpy`

## STEP 2

- Create `launch.json` in `.vscode`
- Add a configuration named `Python: Remote Attach`
- Sometimes you may have to make some small modifications:  `"remoteRoot": "."` -> `"remoteRoot": "${workspaceFolder}"`

## STEP 3

`python3 -m debugpy --listen localhost:5678 paragen/entries/run.py --config train.yaml`

## STEP 4

Press `F5` or click `Python: Remote Attach` to attach to the debug session.
