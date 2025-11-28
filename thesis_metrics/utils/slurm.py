"""SLURM job submission and management utilities"""

import re
import subprocess
from typing import Optional


class SlurmJobManager:
    """Handles SLURM job submission and dependency tracking."""

    def __init__(self):
        self.last_job_id: Optional[int] = None

    def submit(self, cmd: str, dependency: bool = True) -> int:
        """
        Submit a SLURM job with optional dependency on previous job.

        Args:
            cmd: The sbatch command to run (without 'sbatch' prefix)
            dependency: If True, add dependency on last submitted job

        Returns:
            Job ID of the submitted job
        """
        if dependency and self.last_job_id is not None:
            full_cmd = f"sbatch --dependency=afterok:{self.last_job_id} {cmd}"
        else:
            full_cmd = f"sbatch {cmd}"

        output = self._run_cmd(full_cmd)
        self.last_job_id = self._extract_job_id(output)
        return self.last_job_id

    def _run_cmd(self, cmd: str) -> str:
        """Execute a shell command and return stdout."""
        print(f"Running: {cmd}")
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.returncode != 0:
            print(result.stderr)
            raise RuntimeError(f"Command failed: {result.stderr}")
        return result.stdout.strip()

    def _extract_job_id(self, output: str) -> int:
        """Extract job ID from sbatch output."""
        match = re.search(r"Submitted batch job (\d+)", output)
        if not match:
            raise ValueError(f"No job ID found in output: {output}")
        return int(match.group(1))
