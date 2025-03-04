# jax_examples

> Deep Learning examples using the Jax ecosystem of libraries

[![Build Status](https://shawonashraf.visualstudio.com/jax-examples/_apis/build/status%2FShawonAshraf.jax_examples?branchName=main)](https://shawonashraf.visualstudio.com/jax-examples/_build/latest?definitionId=13&branchName=main)

## env setup

> [!IMPORTANT]
> Ensure that you've `uv` installed: [link](https://docs.astral.sh/uv/getting-started/installation/). 

```bash
uv sync
source .venv/bin/activate
```

> [!TIP]
> You can also use the provided devcontainer configuration.
> ```bash
> devcontainer up --workspace-folder .
> ```

> [!CAUTION]
> `uv` doesn't support installing venvs at custom locations so
> if you run the devcontainer after creating a local env, they'll conflict
> due to both venvs trying to exist in the same place. `poetry` has a better solution
> but this project uses uv so anyways.


## run jupyter lab

```bash
# inside the project root
jupyter lab
```
