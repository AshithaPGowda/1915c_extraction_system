# Getting started

Set-up steps for the 1915(c) waiver extraction project, from download to a running app. This is the short version. The full manuals, with screenshots and every feature explained, are the PDF guides in `data/pdf_guides/`: `waiver_adk_guide.pdf` (waiver-level extraction) and `services_adk_guide.pdf` (service-level extraction). Both use the same set-up as below.

> **This tool needs Google Cloud credits or a billing account.** Every extraction calls Google's Gemini models and is charged to your Google Cloud project. Turn on billing (or confirm you have free credits) before your first run, and watch your spend.

## What you need

- A computer with **macOS** (everything was tested here) or Linux. **Not tested on Windows**; steps may fail or need changes.
- **Python 3.10 or newer**, and **git**.
- Your **own** Google account and Google Cloud project, with billing or credits.
- Google Chrome (recommended).

> Work, school or other institutional Google accounts may have restrictions. You may need an administrator to enable billing or specific APIs (such as Vertex AI). That is not covered here: these steps assume your own account and your own project.

## Steps

**1. Download the project**

```bash
git clone <repository-url> 1915c_extraction_agent
cd 1915c_extraction_agent
```

**2. Create a Python environment and install the packages**

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r waiver_level_extraction_agent/requirements.txt
```

**3. Install poppler** (a small tool used to read PDF page images)

```bash
brew install poppler               # macOS (Homebrew)
sudo apt install poppler-utils     # Ubuntu / Debian
```

**4. Install the Google Cloud CLI and sign in**

Install `gcloud` from <https://cloud.google.com/sdk/docs/install>, then:

```bash
gcloud auth application-default login
```

A browser window opens. Choose your Google account and allow access. Do not set `GOOGLE_APPLICATION_CREDENTIALS`.

**5. Get your Google Cloud project ID, and turn on billing and Vertex AI**

In the [Google Cloud console](https://console.cloud.google.com), use the project picker at the top to choose your project and copy its **ID** (not the name or number). Make sure billing is enabled, then turn on the Vertex AI API:

```bash
gcloud services enable aiplatform.googleapis.com --project=YOUR_PROJECT_ID
```

Google's own pages show the console screens: *View and update projects* and *Set up a project and a development environment* at <https://docs.cloud.google.com>.

**6. Create your `.env` file**

```bash
cp waiver_level_extraction_agent/.env.example waiver_level_extraction_agent/.env
```

Open `waiver_level_extraction_agent/.env` and set **both** of these to your project ID (they must be identical):

```
PROJECT_ID=your-project-id
GOOGLE_CLOUD_PROJECT=your-project-id
```

Leave the other lines as they are. Never share or upload `.env`.

> Service-level extraction caches large PDFs in a Google Cloud Storage bucket. If you do not set `CACHE_GCS_BUCKET` in `.env`, a bucket named `<project-id>-waiver-pdf-cache` is created the first time it is needed, which needs permission to create buckets. If you cannot create one, create a bucket yourself and put its name in `CACHE_GCS_BUCKET`.

**7. Start the app**

```bash
adk web . --port 8000
```

Open <http://127.0.0.1:8000> in Chrome, click **Select an app**, and choose `waiver_level_extraction_agent` (waiver-level data) or `services_extraction_agent` (service-level data). Ignore the entry called `data`. Keep the terminal open; press `Ctrl+C` to stop.

## Check that it works

```bash
python3 -c "from waiver_level_extraction_agent.shared_libraries.config_check import check_cloud_config as c; c(); print('config ok')"
gcloud auth application-default print-access-token > /dev/null && echo "sign-in ok"
```

You should see `config ok` and `sign-in ok`, and the app should load in your browser. No money is spent until you ask it to extract a waiver.

If the app refuses to start and says the project is not set or that `PROJECT_ID` and `GOOGLE_CLOUD_PROJECT` differ, fix those two lines in `.env` (step 6).

## Set up with an AI assistant

If you already use an AI assistant, especially one that can run commands on your computer, it can do steps 1 to 3 and 6 to 7 itself. It cannot do the Google parts for you (creating the project, billing, enabling the Vertex AI API, and the sign-in in step 4), so it should stop at those and wait. Paste the prompt below, after replacing the text in square brackets.

```text
You are my setup assistant. Your job: take this project from a fresh download to a working app on my computer, doing everything you can yourself, and stopping to hand me the few steps that only I can do.

THE PROJECT
A tool that extracts data from 1915(c) Medicaid waiver PDFs using Google's Gemini models through Google Cloud (Vertex AI). Repository address: [repository URL, or "already cloned in this folder"]

ABOUT ME
- Operating system: [macOS / Linux / Windows]
- Google Cloud project ID: [your project ID, or "I don't have one yet"]
- I am using my own personal Google account and my own Google Cloud project.

WHAT YOU DO YOURSELF (explain each command first, and ask before anything that installs software or changes system settings)
1. Get the project: git clone the repository (skip if it is already here), then work from the project root folder.
2. Check that Python 3.10 or newer is installed. Create and activate a virtual environment named .venv in the project root.
3. Install the packages: pip install -r waiver_level_extraction_agent/requirements.txt
4. Install poppler for my operating system (macOS: brew install poppler; Debian/Ubuntu: sudo apt install poppler-utils).
5. Install the gcloud CLI if it is missing (https://cloud.google.com/sdk/docs/install).
6. Copy waiver_level_extraction_agent/.env.example to waiver_level_extraction_agent/.env. When I give you my project ID, set BOTH PROJECT_ID and GOOGLE_CLOUD_PROJECT to it. They must be identical. Leave the other settings as they are.

WHAT ONLY I CAN DO (stop, tell me exactly what to click or type, and wait until I say it is done)
A. Create a Google Cloud project, or choose an existing one, and give you its project ID (the ID, not the name or number).
B. Turn on billing, or confirm I have free credits, for that project. Model calls are charged to it.
C. Turn on the Vertex AI API for the project (console API library, or: gcloud services enable aiplatform.googleapis.com --project=<ID>). You may run the command if I ask you to.
D. Sign in with: gcloud auth application-default login (a browser window opens and I complete it myself).

THEN CHECK THAT IT WORKS (report each result)
1. The project settings are consistent. Run this from the project root with the environment active:
   python3 -c "from waiver_level_extraction_agent.shared_libraries.config_check import check_cloud_config as c; c(); print('config ok')"
2. My Google sign-in works (this must not print any token):
   gcloud auth application-default print-access-token > /dev/null && echo "sign-in ok"
3. Start the app: adk web . --port 8000
   Confirm that http://127.0.0.1:8000 loads and that waiver_level_extraction_agent appears in the app list. (The entry called "data" is not an app.) Leave the app running and tell me to open that address in my browser.

RULES
- Do NOT run an extraction, or anything that calls the Gemini model, unless I tell you to. Those calls cost money.
- Never ask me to paste passwords, tokens or key files into this chat. Never write secrets into files. Never commit, upload or print my .env file.
- Do not enable billing or APIs on your own. Do not change anything outside this project folder except installing the tools above.
- If a step fails, show me the exact error, explain it in plain language, and suggest the smallest fix. Common ones: PROJECT_ID and GOOGLE_CLOUD_PROJECT differ (set both to the same value); a 403 or billing error (billing or the Vertex AI API is not enabled, or I signed in with the wrong account); poppler not found (install it and reopen the terminal).
- This project has not been tested on Windows. If something looks Windows-specific, tell me it may not work instead of guessing.
- If you cannot run commands, give me the commands one at a time, wait for me to paste the output, and continue from there.
- When you finish, summarize what you installed and changed, and what is still left for me to do.
```

## Where to go next

- `data/pdf_guides/waiver_adk_guide.pdf`: extract the 163 waiver-level fields in the chat window, review and correct them, extract a custom value, run batches from the command line, and fix common problems.
- `data/pdf_guides/services_adk_guide.pdf`: extract the services in a waiver (with their parent and child structure), review and correct them, extract a custom value about a service, use queue mode, and run the services pipeline from the command line.
