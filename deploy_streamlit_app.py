#!/usr/bin/env python3
"""Automate the deployment of a Streamlit app to GitHub and Streamlit Cloud.

Features:
- Prerequisite checks for Python, pip, git, and required packages
- Option to create or reuse a GitHub repository
- Git initialization, commit, and push automation
- File generation for Streamlit deployment
- Local validation of app startup
- Deployment log with rollback support
- Interactive and silent modes
- Colored console output via colorama
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

from colorama import Fore, Style, init

init(autoreset=True)


class DeploymentError(RuntimeError):
    """Raised when deployment operations fail."""


class DeploymentManager:
    def __init__(self, project_dir: Path, repo_name: str | None = None, github_token: str | None = None,
                 github_username: str | None = None, silent: bool = False, interactive: bool = True,
                 skip_git: bool = False, skip_push: bool = False, skip_local_test: bool = False):
        self.project_dir = project_dir.resolve()
        self.repo_name = repo_name or self.project_dir.name
        self.github_token = github_token
        self.github_username = github_username
        self.silent = silent
        self.interactive = interactive
        self.skip_git = skip_git
        self.skip_push = skip_push
        self.skip_local_test = skip_local_test
        self.log_path = self.project_dir / 'deployment.log'
        self.rollback_log = []

    def log(self, message: str, color: str = Fore.WHITE, level: str = 'INFO'):
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        formatted = f'[{timestamp}] [{level}] {message}'
        if not self.silent:
            print(f'{color}{formatted}{Style.RESET_ALL}')
        with self.log_path.open('a', encoding='utf-8') as fh:
            fh.write(formatted + '\n')

    def run_command(self, command: list[str] | str, check: bool = True, capture_output: bool = True, shell: bool = False):
        """Execute a subprocess command and return the result."""
        if isinstance(command, str):
            command_str = command
        else:
            command_str = ' '.join(command)

        self.log(f'Running: {command_str}', Fore.CYAN)
        try:
            result = subprocess.run(
                command,
                shell=shell,
                check=check,
                capture_output=capture_output,
                text=True,
            )
            if result.stdout and not self.silent:
                print(result.stdout.strip())
            if result.stderr and not self.silent:
                print(result.stderr.strip())
            return result
        except subprocess.CalledProcessError as exc:
            stderr = exc.stderr.strip() if exc.stderr else 'Unknown error'
            stdout = exc.stdout.strip() if exc.stdout else ''
            self.log(f'Command failed: {command_str}', Fore.RED, 'ERROR')
            self.log(stderr or stdout, Fore.RED, 'ERROR')
            if check:
                raise DeploymentError(f'Command failed: {command_str}\n{stderr or stdout}')
            return exc

    def ensure_prerequisites(self):
        """Verify Python, pip, git and key dependencies are installed."""
        self.log('Checking prerequisites...', Fore.YELLOW)

        python_ok = sys.version_info >= (3, 8)
        if not python_ok:
            raise DeploymentError('Python 3.8+ is required.')
        self.log(f'Python version OK: {sys.version}', Fore.GREEN)

        try:
            subprocess.run([sys.executable, '-m', 'pip', '--version'], check=True, capture_output=True, text=True)
            self.log('pip is available.', Fore.GREEN)
        except Exception as exc:
            raise DeploymentError(f'pip verification failed: {exc}')

        git_ok = shutil.which('git') is not None
        if not git_ok:
            raise DeploymentError('git is not installed or not available in PATH.')
        self.log('git is available.', Fore.GREEN)

        required_packages = ['streamlit', 'pandas', 'numpy', 'scikit-learn', 'matplotlib', 'seaborn', 'plotly', 'scipy', 'openpyxl', 'altair', 'colorama']
        missing = []
        for package in required_packages:
            try:
                __import__(package)
            except Exception:
                missing.append(package)

        if missing:
            self.log(f'Missing Python packages: {missing}', Fore.YELLOW)
            if self.interactive:
                install = input('Install missing packages now? [Y/n]: ').strip().lower()
                if install in ('', 'y', 'yes'):
                    self.run_command([sys.executable, '-m', 'pip', 'install', *missing])
                else:
                    raise DeploymentError('Missing required packages. Install them before deployment.')
            else:
                self.run_command([sys.executable, '-m', 'pip', 'install', *missing])

        self.log('Prerequisites check completed successfully.', Fore.GREEN)

    def ensure_git_identity(self):
        """Configure git local user.name/user.email if missing."""
        if self.skip_git:
            return

        name_check = self.run_command(['git', 'config', '--get', 'user.name'], check=False)
        email_check = self.run_command(['git', 'config', '--get', 'user.email'], check=False)

        if name_check.returncode != 0:
            if self.interactive:
                git_name = input('Enter git username/name: ').strip() or 'Deployment Bot'
            else:
                git_name = 'Deployment Bot'
            self.run_command(['git', 'config', 'user.name', git_name])

        if email_check.returncode != 0:
            if self.interactive:
                git_email = input('Enter git email: ').strip() or 'deployment@example.com'
            else:
                git_email = 'deployment@example.com'
            self.run_command(['git', 'config', 'user.email', git_email])

    def check_github_repo(self):
        """Check if a GitHub repository exists and optionally create it."""
        if not self.github_token:
            if self.interactive:
                self.github_token = input('GitHub personal access token (optional if repo already exists): ').strip()
            else:
                raise DeploymentError('A GitHub token is required for automated repository creation in silent mode.')

        if self.github_username is None:
            if self.interactive:
                self.github_username = input('GitHub username: ').strip()
            else:
                raise DeploymentError('GitHub username is required in silent mode.')

        from github import Github

        gh = Github(self.github_token)
        user = gh.get_user()
        try:
            repo = user.get_repo(self.repo_name)
            self.log(f'Repository already exists: {repo.html_url}', Fore.GREEN)
            return repo.html_url
        except Exception:
            self.log(f'Creating new repository: {self.repo_name}', Fore.YELLOW)
            repo = user.create_repo(self.repo_name, private=False)
            self.log(f'Created repository: {repo.html_url}', Fore.GREEN)
            return repo.html_url

    def create_project_files(self):
        """Create required directories and files for deployment."""
        self.log('Preparing project files...', Fore.YELLOW)

        self.project_dir.mkdir(parents=True, exist_ok=True)
        (self.project_dir / '.streamlit').mkdir(exist_ok=True)

        files = {
            'requirements.txt': '''streamlit>=1.28.0
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
seaborn>=0.12.0
plotly>=5.17.0
scipy>=1.11.0
openpyxl>=3.1.0
altair>=5.0.0
colorama>=0.4.6
''',
            '.gitignore': '''__pycache__/
*.pyc
*.pyo
*.pyd
.pytest_cache/
.mypy_cache/
.venv/
venv/
.env
.env.*
.DS_Store
.vscode/
*.log
*.sqlite3
*.xlsx
*.csv
''',
            '.streamlit/config.toml': '''[theme]
base = "light"
primaryColor = "#1f77b4"
backgroundColor = "#ffffff"
secondaryBackgroundColor = "#f4f6f9"
textColor = "#1f2937"
font = "sans serif"

[server]
headless = true
port = 8501
enableCORS = false
enableXsrfProtection = true

[browser]
gatherUsageStats = false
''',
            'README.md': '''# Customer Segmentation Dashboard

A Streamlit dashboard for comparing customer segmentation models.

## Features
- K-Means clustering
- Gaussian Mixture Models
- Hierarchical clustering
- Interactive dashboards
- Exportable outputs

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deployment

This project is designed for deployment on Streamlit Cloud.
''',
            'setup.sh': '''#!/bin/bash
set -e
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py --server.headless true --server.port 8501
''',
            'deployment_checklist.md': '''# Deployment Checklist

- [ ] Python dependencies installed
- [ ] app.py runs locally
- [ ] Git repository initialized
- [ ] GitHub repo created
- [ ] Code pushed to remote
- [ ] Streamlit Cloud project connected
- [ ] Deployment environment configured
- [ ] App loads without errors
''',
            'deployment_badge.md': '''[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/)
''',
        }

        for file_name, content in files.items():
            path = self.project_dir / file_name
            path.write_text(content, encoding='utf-8')
            self.log(f'Created {file_name}', Fore.GREEN)

        # ensure setup.sh executable
        setup_path = self.project_dir / 'setup.sh'
        if setup_path.exists():
            setup_path.chmod(0o755)

        # create placeholder directories
        for directory in ['data', 'artifacts', 'logs']:
            (self.project_dir / directory).mkdir(exist_ok=True)

    def initialize_git(self):
        """Initialize git repository and commit files."""
        if self.skip_git:
            return

        self.log('Initializing git repository...', Fore.YELLOW)
        if not (self.project_dir / '.git').exists():
            self.run_command(['git', '-C', str(self.project_dir), 'init'])
            self.rollback_log.append('git init')

        self.run_command(['git', '-C', str(self.project_dir), 'checkout', '-B', 'main'], check=False)
        self.run_command(['git', '-C', str(self.project_dir), 'add', '.'])
        self.run_command(['git', '-C', str(self.project_dir), 'commit', '-m', 'Initial commit'], check=False)

    def push_to_github(self, repo_url: str | None = None):
        """Add remote and push to GitHub."""
        if self.skip_push:
            return None

        if repo_url is None:
            repo_url = self.check_github_repo()

        self.log(f'Adding remote origin: {repo_url}', Fore.YELLOW)
        self.run_command(['git', '-C', str(self.project_dir), 'remote', 'remove', 'origin'], check=False)
        self.run_command(['git', '-C', str(self.project_dir), 'remote', 'add', 'origin', repo_url])
        self.run_command(['git', '-C', str(self.project_dir), 'push', '-u', 'origin', 'main'])
        self.log(f'Pushed repository to GitHub: {repo_url}', Fore.GREEN)
        return repo_url

    def verify_local_app(self):
        """Run a lightweight local validation of the Streamlit app."""
        if self.skip_local_test:
            return True

        self.log('Running local app validation...', Fore.YELLOW)
        app_path = self.project_dir / 'app.py'
        if not app_path.exists():
            raise DeploymentError('app.py is missing. Create the app before running deployment automation.')

        # syntax validation
        self.run_command([sys.executable, '-m', 'py_compile', str(app_path)])

        # dependency validation
        required = ['streamlit', 'pandas', 'numpy', 'sklearn', 'matplotlib', 'seaborn', 'plotly', 'scipy', 'openpyxl']
        for package in required:
            try:
                __import__(package)
            except Exception as exc:
                raise DeploymentError(f'Missing dependency for validation: {package} ({exc})')

        self.log('Local app validation successful.', Fore.GREEN)
        return True

    def generate_deployment_instructions(self):
        """Create instructions for Streamlit Cloud deployment."""
        content = '''# Streamlit Cloud Deployment Guide

1. Push your code to GitHub.
2. Go to https://share.streamlit.io/.
3. Sign in with your GitHub account.
4. Click `New app`.
5. Select the repository and branch.
6. Set the main file path to `app.py`.
7. Choose Python version 3.11.
8. Click `Deploy`.
9. Wait for build completion.
10. Share your public URL.

## Optional settings
- Set environment variables if needed.
- Add custom domain if available.
- Configure secrets in Streamlit Cloud settings.
'''
        (self.project_dir / 'streamlit_cloud_deployment.md').write_text(content, encoding='utf-8')

    def rollback(self):
        """Attempt a basic rollback of created files and git state."""
        self.log('Rolling back failed deployment...', Fore.RED, 'WARNING')
        for item in reversed(self.rollback_log):
            self.log(f'Rollback step: {item}', Fore.YELLOW)

    def run(self):
        """Run the full deployment automation workflow."""
        self.log('Starting deployment automation...', Fore.GREEN)
        try:
            self.ensure_prerequisites()
            self.ensure_git_identity()
            self.create_project_files()
            self.generate_deployment_instructions()
            self.verify_local_app()

            if not self.skip_git:
                self.initialize_git()
                if not self.skip_push:
                    repo_url = self.push_to_github()
                else:
                    repo_url = None
            else:
                repo_url = None

            self.log('Deployment automation complete.', Fore.GREEN)
            self.log(f'Repository URL: {repo_url or "Not pushed"}', Fore.GREEN)
            return {
                'repo_url': repo_url,
                'project_dir': str(self.project_dir),
                'log_path': str(self.log_path),
            }
        except Exception as exc:
            self.log(f'Deployment failed: {exc}', Fore.RED, 'ERROR')
            self.rollback()
            raise


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='Automate Streamlit app deployment to GitHub and Streamlit Cloud.')
    parser.add_argument('--project-dir', default='.', help='Project directory to deploy.')
    parser.add_argument('--repo-name', default=None, help='GitHub repository name to create/use.')
    parser.add_argument('--github-token', default=None, help='GitHub personal access token.')
    parser.add_argument('--github-username', default=None, help='GitHub username.')
    parser.add_argument('--silent', action='store_true', help='Suppress console output.')
    parser.add_argument('--no-interactive', action='store_true', help='Disable interactive prompts.')
    parser.add_argument('--skip-git', action='store_true', help='Skip git initialization and commit.')
    parser.add_argument('--skip-push', action='store_true', help='Skip pushing to GitHub.')
    parser.add_argument('--skip-local-test', action='store_true', help='Skip local app verification.')
    return parser.parse_args()


def main():
    args = parse_arguments()
    project_dir = Path(args.project_dir).resolve()
    manager = DeploymentManager(
        project_dir=project_dir,
        repo_name=args.repo_name,
        github_token=args.github_token,
        github_username=args.github_username,
        silent=args.silent,
        interactive=not args.no_interactive,
        skip_git=args.skip_git,
        skip_push=args.skip_push,
        skip_local_test=args.skip_local_test,
    )

    try:
        result = manager.run()
        if not args.silent:
            print(Fore.GREEN + '\nDeployment summary:')
            print(json.dumps(result, indent=2))
    except DeploymentError as exc:
        print(Fore.RED + f'ERROR: {exc}')
        sys.exit(1)
    except Exception as exc:
        print(Fore.RED + f'Unexpected failure: {exc}')
        sys.exit(1)


if __name__ == '__main__':
    main()
