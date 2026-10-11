# Contributions Guide

This guide explains how each team member can clone the project, make changes in VSCode, keep the repository up to date, and record updates in the changelog.

---

##  1. Getting Started

### Step 1 – Accept the repository invitation
You’ll receive an invite to join the project on GitHub.  
Accept it, then you’ll have permission to push and pull.

### Step 2 – Clone the project into VS Code

#### Option A — Using the VS Code interface (recommended)
1. Open **VS Code**.
2. Click the **Source Control** icon in the left sidebar  
   *(or press `Ctrl + Shift + G`)*.
3. Click **Clone Repository**.
4. Paste the repository URL, for example:  
   https://github.com/JonathanMa03/innovative-droBVAR
5. Choose a local folder (e.g., `~/Documents/`).
6. Click **Open** when prompted.

VS Code will download the repository and open it as your workspace.

#### Option B — Using the terminal
Open the VS Code terminal (**Terminal → New Terminal**) and run:

```bash
cd ~/Documents
git clone https://github.com/JonathanMa03/project-name
cd innovative-droBVAR
code .
```
The folder will open in VS Code and you can manage Git using the Source Control panel.

---

## 2. Keeping Your Copy Up-to-Date

Before starting any work, **always pull the latest version**:

```bash
git pull origin main
```

### Using VS Code
1.	Open the Source Control panel.
2.	Click the … menu → Pull (or click the sync icon if visible).

This prevents merge conflicts and ensures you’re working on the newest files.

---
## 3. Branching and Collaboration

### Why Use Branches?

Branching allows everyone to work on new features, analyses, or documentation **without changing the stable main branch**.

**In short:**
- Keeps `main` clean and functional.  
- Prevents overwriting teammates’ work.  
- Makes it easy to review and merge code safely.  

Think of branches as “temporary workspaces” for specific tasks (e.g., `feature/eda-update`, `feature/model-improvement`).

---

### How to Create and Use a Branch

#### Option A — In VS Code (GUI)
1. Click the current branch name in the **bottom-left corner** of VS Code (likely `main`).
2. Select **Create new branch**.
3. Enter a descriptive name, for example: `model-update`.
4. Make your changes, then commit and push as usual.

VS Code will automatically track and publish the new branch.

#### Option B — In the Terminal
```bash
# Make sure main is up-to-date
git checkout main
git pull origin main

# Create and switch to a new branch
git checkout -b model-update
```

When you finish your work:

```bash
git add .
git commit -m "feat: add model diagnostic plots"
git push origin model-update
```

Then:

1. Go to GitHub.
2. Open a **Pull Request (PR)**.
3. Request a review from teammates.
4. Merge into `main` once approved.

This allows others to review changes and helps prevent mistakes.

### Getting updates:

If teammates have updated main, you can sync those changes into your branch:

```bash
git checkout main
git pull origin main
git checkout model-update
git merge main
```

---

## 4. Making and Committing Changes

1. Save your modified files (`.py`, `.ipynb`, `.md`, etc.).
2. Open the **Source Control** panel in VS Code.
3. Stage your changes:
   - Click the **+** next to each file, or select **Stage All Changes**.
4. Enter a short, clear commit message.
5. Click **Commit** ✔.
6. Click **Push** (or **Sync Changes**) to upload your changes.
7. Verify updates on GitHub.

### If You Create a New File or Folder

If you add a new file (for example, `src/new_analysis.py`) or a new folder (like `data/cleaned/` or `figures/`):

1. **Make sure it appears in the Source Control panel.**  
   - If you don’t see it, click the **Refresh** icon.
2. **Stage it manually** by clicking the **+** next to the file.
3. **Commit** it with a clear message describing the addition.
4. **Push** your changes so the new files are available to the team.
5. If the new folder doesn’t appear because it’s empty, add a small placeholder file so Git tracks it:

```bash
touch figures/.keep
```

Tip: Always check that your `.gitignore` file isn’t excluding your new folder or file type.

---

##  5. Updating the Change log

After making changes:

1.	Open docs/CHANGELOG.md.
2.	Under "Added" or "Changed", add a bullet for your update:

```markdown
### Added
- JM 11/7: Initial Repository Creation, added directories for docs and data.
- JM 11/8: Added data files and Project Initialization

### Changed
- JM 11/8: Changed formatting and instructions in Contributions.md.
```

**TL;DR Workflow**

- Pull the latest changes before starting work.
- For major changes: create a new branch.
- For small fixes: you may commit directly to `main` (if your team agrees).
- Make your changes and save your files.
- Open **Source Control** → stage changes → **Commit** → **Push**.
- On GitHub, click **Compare & pull request** and briefly describe what you changed.
- After your PR is approved and merged, pull the latest updates before continuing.
