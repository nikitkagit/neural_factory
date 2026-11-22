import re
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Optional


class DangerLevel(Enum):
    """Severity level of a detected risk."""
    MEDIUM = auto()
    HIGH = auto()
    CRITICAL = auto()


@dataclass
class RiskItem:
    """Represents a single detected risk in the script."""
    level: DangerLevel
    line_number: int
    command: str
    description: str


@dataclass
class ValidationResult:
    """Result of the bash script validation."""
    is_safe: bool
    risks: List[RiskItem] = field(default_factory=list)
    blocked_reason: Optional[str] = None


class BashScriptValidator:
    """
    Static analyzer for Bash scripts to detect potentially dangerous operations.
    Implements fail-safe principle: parsing errors are treated as CRITICAL.
    """

    # Patterns for CRITICAL risks
    CRITICAL_PATTERNS = [
        (re.compile(r'rm\s+(-[a-zA-Z]*r[a-zA-Z]*f[a-zA-Z]*|-[a-zA-Z]*f[a-zA-Z]*r[a-zA-Z]*)\s+/(?:\s|$)'), "Recursive removal of root directory"),
        (re.compile(r'mkfs\b'), "Filesystem formatting detected"),
        (re.compile(r'dd\s+.*of=/dev/[sh]d[a-z]'), "Direct disk writing detected"),
        (re.compile(r'chmod\s+777\s+/etc/shadow'), "Critical permission change on shadow file"),
        (re.compile(r'>\s*/dev/sd'), "Redirection to system device"),
    ]

    # Patterns for HIGH risks
    HIGH_PATTERNS = [
        (re.compile(r'\bsudo\b'), "Privilege escalation via sudo"),
        (re.compile(r'\bapt\s+install\b'), "System package installation"),
        (re.compile(r'\bpip\s+install\b'), "Python package installation"),
        (re.compile(r'\bcurl\b|\bwget\b'), "External network request"),
    ]

    # Patterns for MEDIUM risks
    MEDIUM_PATTERNS = [
        (re.compile(r'cat\s+/etc/passwd'), "Reading sensitive system file"),
        (re.compile(r'cat\s+~/.ssh/id_rsa'), "Reading SSH private key"),
        (re.compile(r'\bnohup\b|\&$'), "Background process execution"),
    ]

    def validate(self, script: str) -> ValidationResult:
        """
        Validates a bash script string for dangerous operations.
        
        :param script: The bash script content.
        :return: ValidationResult indicating safety and detected risks.
        """
        risks: List[RiskItem] = []
        
        try:
            lines = script.splitlines()
            for i, line in enumerate(lines, 1):
                stripped = line.strip()
                if not stripped or stripped.startswith('#'):
                    continue
                
                self._check_patterns(stripped, i, risks)
                
        except Exception as e:
            return ValidationResult(
                is_safe=False,
                blocked_reason=f"Parser error: {str(e)}",
                risks=[RiskItem(DangerLevel.CRITICAL, 0, "PARSER_ERROR", str(e))]
            )

        is_safe = len([r for r in risks if r.level == DangerLevel.CRITICAL]) == 0
        blocked_reason = "Critical risks detected" if not is_safe else None
        
        return ValidationResult(
            is_safe=is_safe,
            risks=risks,
            blocked_reason=blocked_reason
        )

    def _check_patterns(self, line: str, line_num: int, risks: List[RiskItem]) -> None:
        """Checks a single line against all risk patterns."""
        for pattern, description in self.CRITICAL_PATTERNS:
            if pattern.search(line):
                risks.append(RiskItem(DangerLevel.CRITICAL, line_num, line, description))

        for pattern, description in self.HIGH_PATTERNS:
            if pattern.search(line):
                risks.append(RiskItem(DangerLevel.HIGH, line_num, line, description))

        for pattern, description in self.MEDIUM_PATTERNS:
            if pattern.search(line):
                risks.append(RiskItem(DangerLevel.MEDIUM, line_num, line, description))