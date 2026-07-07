## Security

NVIDIA is dedicated to the security and trust of our software products and services, including all source code repositories managed through our organization.

If you need to report a security issue, please use the appropriate contact points outlined below. **Please do not report security vulnerabilities through GitHub.** If a potential security issue is inadvertently reported via a public issue or pull request, NVIDIA maintainers may limit public discussion and redirect the reporter to the appropriate private disclosure channels.

## Reporting Potential Security Vulnerability in an NVIDIA Product

To report a potential security vulnerability in any NVIDIA product:
- Web: [Security Vulnerability Submission Form](https://www.nvidia.com/object/submit-security-vulnerability.html)
- E-Mail: psirt@nvidia.com
    - We encourage you to use the following PGP key for secure email communication: [NVIDIA public PGP Key for communication](https://www.nvidia.com/en-us/security/pgp-key)
    - Please include the following information:
   	 - Product/Driver name and version/branch that contains the vulnerability
     - Type of vulnerability (code execution, denial of service, buffer overflow, etc.)
   	 - Instructions to reproduce the vulnerability
   	 - Proof-of-concept or exploit code
   	 - Potential impact of the vulnerability, including how an attacker could exploit the vulnerability

While NVIDIA currently does not have a bug bounty program, we do offer acknowledgement when an externally reported security issue is addressed under our coordinated vulnerability disclosure policy. Please visit our [Product Security Incident Response Team (PSIRT)](https://www.nvidia.com/en-us/security/psirt-policies/) policies page for more information.

## NVIDIA Product Security

For all security-related concerns, please visit NVIDIA's Product Security portal at https://www.nvidia.com/en-us/security

## Reporting Details for Skill Evaluator

When reporting a vulnerability in Skill Evaluator, include as much of the
following information as you can safely provide:

- Product name: Skill Evaluator
- Affected version, branch, tag, or commit
- Vulnerability type and potential impact
- Reproduction steps
- Proof-of-concept details, if available
- Relevant configuration, command line, or environment details

## Supported Versions

Security updates are provided for the latest public release of Skill Evaluator
unless a release note states otherwise.

For unreleased branches or pre-release snapshots, include the affected branch,
tag, or commit in your report so maintainers can reproduce the issue against
the correct source state.

## Filesystem Staging Boundary

Harbor staging rejects non-ignored links, special files, hard-linked files, and
source device crossings, and it keeps private staging roots restricted until
their final publication step. Its repeated validation passes are best-effort
detection for incidental filesystem changes; they do not create a coherent
filesystem snapshot.

All concurrent mutation by another process running as the same operating-system
user is outside the staging security guarantee. This includes one-shot changes
to source, destination, private stage, rollback, or reserve paths after a
validation sample. Callers that include hostile same-user processes in their
threat model must provide exclusive filesystem ownership, process/filesystem
isolation, or snapshot support before invoking evaluation.

## Public Issues

General bugs, documentation issues, and feature requests may be filed through
the project's normal issue tracker. Security vulnerabilities must use the
private reporting channels above — do not file public issues, discussions, or
pull requests for security reports.
