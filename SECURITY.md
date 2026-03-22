# Security Policy

## Supported Versions

| Version | Supported          |
|---------|--------------------|
| 4.x     | ✅ Active support  |
| < 4.0   | ❌ No longer supported |

## Reporting a Vulnerability

**Do NOT open a public issue for security vulnerabilities.**

Instead, please email: **[stmallela.us01@gmail.com]**

Include:
- Description of the vulnerability
- Steps to reproduce
- Impact assessment
- Suggested fix (if any)

You will receive a response within **48 hours** acknowledging receipt, and a detailed response within **7 days** with next steps.

## Scope

The following are in scope:
- Authentication/authorization bypass
- SQL injection or command injection
- Exposure of PHI/PII or HIPAA-protected data
- Credential leakage in code or logs
- Insecure deserialization
- Dependency vulnerabilities (critical/high CVEs)

## HIPAA Considerations

HERA processes clinical data. Any vulnerability that could lead to unauthorized access, modification, or disclosure of Protected Health Information (PHI) is treated as **critical severity**.

## Disclosure Policy

- We follow coordinated disclosure (90-day window)
- Credit will be given to reporters unless they prefer anonymity
- We will publish a security advisory for confirmed vulnerabilities
