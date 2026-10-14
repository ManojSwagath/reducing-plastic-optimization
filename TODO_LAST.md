# TODO LAST

## Deferred Finalization Task
Implement cross-device reproducibility packaging after core research pipeline is stable.

### Scope
1. Add Docker-based runtime with pinned Python and pinned dependencies.
2. Add a single Windows batch entrypoint for one-command execution.
3. Update documentation with Docker-first run flow.

### Planned Files
1. Dockerfile
2. docker-compose.yml
3. run.bat
4. README.md section update

### Acceptance Criteria
1. New machine can run notebooks with one command.
2. No manual venv or pip setup required.
3. Identical behavior across devices.

### Suggested Command
run.bat notebook
