# monorepo-skeleton

## Purpose

Establish a minimal **npm + Nx** repository with strict TypeScript, layered package stubs, and root smoke commands so feature work can attach to real projects without redefining repo shape. **`packages/*` manifests omit internal dependency lists**; TypeScript resolves **`@llm-transformer-diy/*`** via **`tsconfig.base.json`** `paths`. **`apps/cli`** alone declares **`file:`** links to local libraries plus the npm packages the runnable CLI needs (`glob`, `webgpu`, wolf-tui, etc.). No npm workspaces.

## Requirements

### Requirement: Repository links local packages without npm workspaces

The repository SHALL use **`file:`** dependencies only on **`apps/cli`** (pointing at **`packages/*`**) where npm must materialize links. Library **`package.json`** files under **`packages/`** SHALL NOT list **`@llm-transformer-diy/*`** dependencies; those edges are expressed in **`tsconfig.base.json`** `paths`. It MUST NOT rely on npm **`workspaces`** or **`pnpm-workspace.yaml`** for name resolution.

#### Scenario: Local packages resolve from the repository root

- **WHEN** a maintainer runs **`npm install`** from the repository root using the documented package manager major version
- **THEN** **`apps/cli`** (linked from the root via **`file:`**) resolves **`packages/*`** through **`file:`** references and installs the npm packages declared on the CLI (and hoisted types via the root **`devDependencies`** as documented)

---

### Requirement: Nx configures each distinct buildable workspace unit

The repository SHALL contain Nx workspace configuration identifying every `apps/*` and `packages/*` project declared in design (no orphan directories without matching Nx project metadata). Each listed project MUST declare at minimum a build target (even if trivial) reachable from `nx`.

#### Scenario: Build graph is discoverable

- **WHEN** a maintainer runs Nx graph or listprojects for this repository
- **THEN** each canonical package placeholder and the CLI app appears as distinct projects consistent with repository layout documentation

---

### Requirement: Canonical layer packages exist only as compilable stubs

For each layered library named in `docs/project.md` under `packages/` (`gpu`, `tensor`, `autograd`, `tokenizer`, `model`, `training`, `inference`, `io`), the codebase SHALL contain a matching package folder with TypeScript compilation enabled and dependency edges that respect layering (a package MUST NOT declare a dependency on a layer above itself). Packages MUST NOT introduce ML framework or hidden-compute prohibited dependencies listed in project documentation.

#### Scenario: Dependency direction forbids inverted edges

- **WHEN** dependency declarations for workspace packages are reviewed
- **THEN** there is no `package.json` dependency pointing from `gpu` to `tensor` or from `tensor` to `autograd`; lower layers MUST NOT reference higher layers

#### Scenario: Forbidden dependency classes are absent

- **WHEN** root and workspace manifests are inspected for forbidden packages from project documentation (e.g. TensorFlow.js, ONNX runtime, transformers, external autograd stacks)
- **THEN** none of those packages appear in dependency lists for this foundation change

---

### Requirement: Applications use Node-compatible TypeScript

The repository SHALL compile TypeScript in strict mode for workspace projects using CommonJS (`module: commonjs`) and **`esModuleInterop`** suitable for Node.js runtime described in project documentation.

#### Scenario: Strict module interoperability

- **WHEN** the root or project TypeScript configs are inspected
- **THEN** `"strict"` (or equivalent stricter presets) is enabled project-wide without disabling critical checks purely to satisfy stubs

---

### Requirement: Baseline verification commands exist at the workspace root

The root `package.json` SHALL expose scripts implementing at minimum a **lint**, **test**, and **build-all** invocation that execute successfully against stub packages (commands MAY execute no-op suites but MUST exit zero when the repository is correctly initialized).

#### Scenario: Maintainer smoke verification

- **WHEN** a maintainer clones the repo, installs dependencies once, and runs the documented smoke scripts in order lint then test then build
- **THEN** each command completes with exit code zero without requiring GPU hardware or transformer training assets
