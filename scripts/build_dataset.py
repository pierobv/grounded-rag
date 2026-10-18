"""Monta evals/dataset.jsonl a partir dos candidatos gerados + perguntas escritas à mão.

Uso:  uv run python scripts/build_dataset.py
Entrada: evals/candidates.jsonl (gerado por scripts/generate_questions.py)

Este arquivo É o registro da curadoria: para cada candidato aprovado, qual pergunta foi
escolhida (exata ou paráfrase), como foi reescrita e quais fatos-chave valem.
Candidatos ausentes de CURATED foram descartados (motivos em DROPPED).
"""

import json
from pathlib import Path

CANDIDATES = Path("evals/candidates.jsonl")
OUT = Path("evals/dataset.jsonl")

# índice do candidato -> (tipo, pergunta final, fatos-chave)
# Pergunta None = usar a do candidato sem mudanças (a exata ou a paráfrase, conforme o tipo).
CURATED: dict[int, tuple[str, str | None, list[str]]] = {
    0: ("exact", None, ["high risk areas"]),
    1: ("paraphrase", None, ["CsrfViewMiddleware"]),
    2: ("paraphrase", None, ["Cross Site Scripting"]),
    4: (
        "exact",
        "What DNS record can restrict which certificate authorities can issue certificates for a domain?",
        ["CAA"],
    ),
    6: ("paraphrase", None, ["callback identifiers"]),
    7: (
        "paraphrase",
        "Which file should a Rails developer check to ensure that only intended methods and controllers are accessible?",
        ["config/routes.rb"],
    ),
    9: (
        "exact",
        "What is a common failure mode in NoSQL systems where untrusted input leads to unauthorized database access?",
        ["NoSQL Injection"],
    ),
    10: (
        "exact",
        "Which kinds of DOM clobbering can Content Security Policy stop, and which can it not stop?",
        [],
    ),
    11: (
        "paraphrase",
        "How do you set up a catch-all error handler in a Spring application?",
        ["@ExceptionHandler"],
    ),
    12: (
        "exact",
        "Which Mongoose plugin can prevent mass assignment attacks in Node.js?",
        ["mongoose-mass-assign"],
    ),
    13: (
        "paraphrase",
        "What do attackers do to check whether stolen credit card details are still valid?",
        ["Carding"],
    ),
    14: (
        "exact",
        "What is an example of an allowlist regular expression for validating OS command arguments?",
        ["^[a-z0-9]{3,10}$"],
    ),
    16: (
        "exact",
        "What type of vulnerability lets an attacker redirect users by modifying untrusted input to a URL?",
        ["redirect"],
    ),
    18: ("paraphrase", None, ["write your own cryptographic functions"]),
    19: (
        "exact",
        "What steps should be taken when recovering a potentially compromised account?",
        ["suspend or invalidate authenticators"],
    ),
    20: (
        "paraphrase",
        "How can a Rails developer make sure a project lookup only returns projects the current user is allowed to see?",
        ["current_user.projects.find"],
    ),
    23: (
        "exact",
        "When resolving JWT verification keys from header parameters, how do you avoid trusting an attacker-controlled key?",
        ["chain of trust"],
    ),
    24: (
        "paraphrase",
        "What details are recorded when a user's account is locked after too many login attempts?",
        ["maxretries"],
    ),
    25: (
        "paraphrase",
        "How can cookie scoping limit the impact of a subdomain takeover?",
        ["parent domain"],
    ),
    27: (
        "exact",
        "How should developers prevent injection when using ORMs and ODMs in GraphQL?",
        ["ORM injection"],
    ),
    30: ("exact", "Which special characters must be escaped in LDAP search filters?", []),
    31: (
        "paraphrase",
        "How might an attacker's attempt to fill up log files affect the system?",
        ["disk space"],
    ),
    32: (
        "exact",
        "What tools can be used for static analysis of Infrastructure as Code?",
        ["Snyk"],
    ),
    34: (
        "exact",
        "What information should a public vulnerability disclosure include once patched versions are available?",
        ["advisory"],
    ),
    36: ("paraphrase", None, ["absolute timeout"]),
    38: ("paraphrase", None, ["phishing"]),
    39: (
        "exact",
        "How do you prevent server-side template injection when users can write templates in a CMS theme?",
        ["sandbox"],
    ),
    41: (
        "exact",
        "Which threat modeling technique applies six threat categories to components, data flows and trust boundaries?",
        ["STRIDE"],
    ),
    42: ("paraphrase", None, ["DOMPurify"]),
    43: (
        "paraphrase",
        "How can I protect my AI inference API from being misused and keep costs under control?",
        ["spend limits"],
    ),
    44: (
        "paraphrase",
        "What methods can check user inputs before they reach a language model to prevent prompt injection?",
        [],
    ),
    45: (
        "exact",
        "Which HTTP status code can be used to cancel a navigation in an onBeforeUnload handler?",
        ["204"],
    ),
    46: (
        "exact",
        "What are the two 2FA modes supported by npm?",
        ["Authorization-only", "Authorization and write-mode"],
    ),
    47: (
        "exact",
        "How many open XML elements were used in the coercive parsing DoS example against Firefox?",
        ["30,000"],
    ),
    48: (
        "paraphrase",
        "Which devices or systems can protect the private keys used in FIDO authentication?",
        ["Secure Enclave"],
    ),
    49: (
        "exact",
        "Which authentication policy should sensitive App Intents use in iOS 16+ so the device must be unlocked?",
        ["requiresLocalDeviceAuthentication"],
    ),
    50: ("exact", "What properties should WebAuthn passkey challenges have?", ["16 bytes"]),
    51: ("exact", "Why use multi-stage builds in Node.js Docker images?", []),
    52: (
        "exact",
        "What should be done during the preparation phase of virtual patching?",
        ["Pre-Authorization"],
    ),
    53: ("exact", "What are the three main categories of DoS attacks?", ["volumetric"]),
    55: (
        "exact",
        "What are the three service models of the cloud shared responsibility model?",
        ["IaaS", "PaaS", "SaaS"],
    ),
    56: ("paraphrase", "How can I isolate my database from other servers?", ["localhost"]),
    57: ("exact", "What is the default maximum receive message size in grpc-go?", ["4 MiB"]),
    58: (
        "exact",
        "Which protocol protects drones against Wi-Fi deauthentication attacks?",
        ["802.11w"],
    ),
    59: (
        "exact",
        "What should be included in a web cache key so responses are stored correctly?",
        ["query parameters"],
    ),
    60: (
        "paraphrase",
        "Which system handles user identities and authentication in OAuth2, SAML or OIDC?",
        ["Identity Provider"],
    ),
    63: (
        "exact",
        "How do you disable XML external entity support in ColdFusion XmlParse()?",
        ["ALLOWEXTERNALENTITIES"],
    ),
    64: ("paraphrase", "How should my AWS Lambda function get its secrets, and why?", ["runtime"]),
    67: (
        "exact",
        "Which HTTP header controls whether search engines index a resource?",
        ["X-Robots-Tag"],
    ),
    70: (
        "exact",
        "How can I make WebSocket connections safe in a Node.js app using the ws library?",
        ["maxPayload"],
    ),
    71: (
        "exact",
        "What are the two types of third-party vendor JavaScript tags?",
        ["Analytic tags"],
    ),
    73: (
        "paraphrase",
        "How does a helper container work with the main app to deliver secrets in Kubernetes?",
        ["sidecar"],
    ),
    74: (
        "exact",
        "Why should GitHub Actions runners be segregated using runner groups and labels?",
        ["high-privilege runners"],
    ),
    75: (
        "exact",
        "Which Bean Validation annotation checks that a number has a given count of integer and fractional digits?",
        ["@Digits"],
    ),
    76: (
        "exact",
        "What does NIST SP 800-63B say about prompting users for specific types of information for account recovery?",
        ["SHALL NOT"],
    ),
    78: (
        "paraphrase",
        "Why can an HTTP interceptor fail to stop XSS when data comes from databases or other internal services?",
        ["persistent XSS"],
    ),
}

DROPPED = {
    3: "vaga (qualquer guia de SBOM responde)",
    5: "fatos-chave longos demais; pergunta de checklist",
    8: "ambígua: SameSite também está em Session Management",
    15: "vaga",
    17: "ambígua: sandbox de iframe também está em HTML5 Security",
    21: "ambígua: proteção de logs também está em Logging",
    22: "vaga",
    26: "ambígua com Authorization Testing Automation",
    28: "vaga",
    29: "duas perguntas em uma",
    33: "pergunta sobre um exemplo de código, não sobre a recomendação",
    35: "vaga",
    37: "vaga",
    40: "redação confusa",
    54: "ambígua: existe um cheat sheet só de Prototype Pollution",
    61: "fato-chave sem sentido",
    62: "sobre um diagrama em ASCII",
    65: "fato-chave não responde à pergunta",
    66: "meta (sobre o próprio cheat sheet)",
    68: "ambígua com Authorization Regression Testing",
    69: "vaga",
    72: "fatos-chave longos; pergunta de checklist",
    77: "pergunta sobre trecho de código",
    79: "duas perguntas em uma; fatos-chave longos",
}

# Traduções para português de perguntas aprovadas: mesmos rótulos, mede o efeito do idioma.
PORTUGUESE: dict[int, str] = {
    1: "Que parte do código preciso adicionar para proteger uma aplicação Django contra CSRF?",
    4: "Qual registro DNS restringe quais autoridades certificadoras podem emitir certificados para um domínio?",
    7: "Qual arquivo um desenvolvedor Rails deve conferir para garantir que só os métodos e controllers pretendidos estão acessíveis?",
    12: "Qual plugin do Mongoose evita ataques de mass assignment em Node.js?",
    13: "O que atacantes fazem para testar se dados de cartão de crédito roubados ainda são válidos?",
    19: "Que passos devem ser seguidos ao recuperar uma conta possivelmente comprometida?",
    24: "Quais detalhes são registrados quando a conta de um usuário é bloqueada após muitas tentativas de login?",
    31: "Como a tentativa de um atacante de encher os arquivos de log pode afetar o sistema?",
    38: "Que objetivos maliciosos um atacante pode ter com os arquivos que envia por upload?",
    41: "Qual técnica de modelagem de ameaças aplica seis categorias de ameaça a componentes, fluxos de dados e fronteiras de confiança?",
    45: "Qual código de status HTTP pode ser usado para cancelar uma navegação em um handler de onBeforeUnload?",
    47: "Quantos elementos XML abertos foram usados no exemplo de DoS por parsing coercivo contra o Firefox?",
    55: "Quais são os três modelos de serviço do modelo de responsabilidade compartilhada na nuvem?",
    57: "Qual é o tamanho máximo padrão de mensagem recebida no grpc-go?",
    67: "Qual header HTTP controla se mecanismos de busca indexam um recurso?",
}

PW = "owasp/Password_Storage_Cheat_Sheet.md"
SESSION = "owasp/Session_Management_Cheat_Sheet.md"
CLICK = "owasp/Clickjacking_Defense_Cheat_Sheet.md"
CSRF = "owasp/Cross-Site_Request_Forgery_Prevention_Cheat_Sheet.md"
CSP = "owasp/Content_Security_Policy_Cheat_Sheet.md"
UPLOAD = "owasp/File_Upload_Cheat_Sheet.md"
JWT = "owasp/JSON_Web_Token_Cheat_Sheet.md"
LOG = "owasp/Logging_Cheat_Sheet.md"


def sec(source: str, *path: str) -> dict:
    title = {
        PW: "Password Storage Cheat Sheet",
        SESSION: "Session Management Cheat Sheet",
        CLICK: "Clickjacking Defense Cheat Sheet",
        CSRF: "Cross-Site Request Forgery Prevention Cheat Sheet",
        CSP: "Content Security Policy Cheat Sheet",
        UPLOAD: "File Upload Cheat Sheet",
        JWT: "JSON Web Token Cheat Sheet",
        LOG: "Logging Cheat Sheet",
    }[source]
    return {"source": source, "section": " › ".join((title, *path))}


# Perguntas cuja resposta exige juntar duas ou mais seções (todas rotuladas).
MULTI = [
    (
        "What is the difference between a salt and a pepper when storing passwords?",
        [
            sec(PW, "Methods for Enhancing Password Storage", "Salting"),
            sec(PW, "Methods for Enhancing Password Storage", "Peppering"),
        ],
        ["salt", "pepper"],
    ),
    (
        "Which cookie attributes should be set on session ID cookies to protect them?",
        [
            sec(SESSION, "Cookies", "Secure Attribute"),
            sec(SESSION, "Cookies", "HttpOnly Attribute"),
            sec(SESSION, "Cookies", "SameSite Attribute"),
        ],
        ["Secure", "HttpOnly", "SameSite"],
    ),
    (
        "What are the two header-based defenses against clickjacking and how do they differ?",
        [
            sec(CLICK, "Defending with Content Security Policy (CSP) frame-ancestors directive"),
            sec(CLICK, "Defending with X-Frame-Options Response Headers"),
        ],
        ["frame-ancestors", "X-Frame-Options"],
    ),
    (
        "How does the synchronizer token pattern compare with the double-submit cookie pattern for CSRF?",
        [
            sec(CSRF, "Token-Based Mitigation", "Synchronizer Token Pattern"),
            sec(
                CSRF, "Token-Based Mitigation", "ALTERNATIVE: Using A Double-Submit Cookie Pattern"
            ),
        ],
        ["synchronizer", "double-submit"],
    ),
    (
        "How do nonce-based and hash-based strict CSP policies differ?",
        [sec(CSP, "Strict CSP", "Nonce based"), sec(CSP, "Strict CSP", "Hashes")],
        ["nonce", "hash"],
    ),
    (
        "Should uploaded files be validated by extension, by Content-Type or by file signature?",
        [
            sec(UPLOAD, "File Upload Protection", "Extension Validation"),
            sec(UPLOAD, "File Upload Protection", "Content-Type Validation"),
            sec(UPLOAD, "File Upload Protection", "File Signature Validation"),
        ],
        ["extension", "Content-Type", "signature"],
    ),
    (
        "Which JWT attacks involve confusing the key type or trusting keys referenced in the token header?",
        [
            sec(JWT, "Threats on JWTs", "Key type confusion"),
            sec(JWT, "Threats on JWTs", "Trusting key material named in the token header"),
        ],
        [],
    ),
    (
        "When should a session expire automatically, and how should logout invalidate it?",
        [
            sec(SESSION, "Session Expiration", "Automatic Session Expiration"),
            sec(SESSION, "Session Expiration", "Manual Session Expiration"),
        ],
        ["timeout"],
    ),
    (
        "Besides Argon2id, which password hashing algorithms are recommended and with which parameters?",
        [
            sec(PW, "Password Hashing Algorithms", "scrypt"),
            sec(PW, "Password Hashing Algorithms", "bcrypt"),
            sec(PW, "Password Hashing Algorithms", "PBKDF2"),
        ],
        ["scrypt", "bcrypt", "PBKDF2"],
    ),
    (
        "What data should never be written to application logs, and how should the logs themselves be protected?",
        [
            sec(LOG, "Design, implementation, and testing", "Data to exclude"),
            sec(LOG, "Deployment and operation", "Protection"),
        ],
        [],
    ),
]

# Fora do corpus. Conferidas com grep: os termos-chave não aparecem nos cheat sheets.
# Metade é de segurança/TI "vizinha" ao corpus (mais difícil: a busca acha algo parecido).
UNANSWERABLE = [
    ("en", "What is the capital of France?"),
    ("en", "Who wrote Hamlet?"),
    ("en", "How many moons does Jupiter have?"),
    ("en", "Who won the 2018 FIFA World Cup?"),
    ("en", "What is the boiling point of water in Fahrenheit?"),
    ("en", "How do I bake sourdough bread?"),
    ("en", "What is the average time complexity of quicksort?"),
    ("en", "How do I reverse a linked list in Python?"),
    ("en", "What is the CVSS score of Log4Shell (CVE-2021-44228)?"),
    ("en", "How do I configure an access control list on a Cisco ASA firewall?"),
    ("en", "What is the default port of PostgreSQL?"),
    ("en", "How much does a Burp Suite Professional license cost?"),
    ("en", "Who is the executive director of the OWASP Foundation?"),
    ("en", "How do I renew a Let's Encrypt certificate with certbot?"),
    ("en", "When was Kubernetes 1.30 released?"),
    ("en", "How many records were exposed in the Equifax breach?"),
    ("en", "Who invented the RSA algorithm?"),
    ("en", "What is the minimum RAM required to install Windows 11?"),
    ("pt", "Qual é a capital da Austrália?"),
    ("pt", "Como configuro uma VPN em um roteador TP-Link?"),
]


def main() -> None:
    candidates = [json.loads(line) for line in CANDIDATES.read_text(encoding="utf-8").splitlines()]
    overlap = CURATED.keys() & DROPPED.keys()
    missing = set(range(len(candidates))) - CURATED.keys() - DROPPED.keys()
    if overlap or missing:
        raise SystemExit(f"Curadoria incompleta: sobreposição {overlap}, sem decisão {missing}")

    rows = []
    for i, (qtype, text, facts) in CURATED.items():
        c = candidates[i]
        question = text or c["exact_question" if qtype == "exact" else "paraphrase_question"]
        relevant = [{"source": c["source"], "section": c["section"]}]
        rows.append(
            {
                "id": f"g{i:02d}",
                "question": question,
                "type": qtype,
                "lang": "en",
                "answerable": True,
                "relevant": relevant,
                "key_facts": facts,
                "origin": "generated",
            }
        )
        if i in PORTUGUESE:
            rows.append(
                {
                    "id": f"pt-g{i:02d}",
                    "question": PORTUGUESE[i],
                    "type": qtype,
                    "lang": "pt",
                    "answerable": True,
                    "relevant": relevant,
                    "key_facts": facts,
                    "origin": "handwritten",
                }
            )
    for n, (question, relevant, facts) in enumerate(MULTI, start=1):
        rows.append(
            {
                "id": f"m{n:02d}",
                "question": question,
                "type": "multi",
                "lang": "en",
                "answerable": True,
                "relevant": relevant,
                "key_facts": facts,
                "origin": "handwritten",
            }
        )
    for n, (lang, question) in enumerate(UNANSWERABLE, start=1):
        rows.append(
            {
                "id": f"u{n:02d}",
                "question": question,
                "type": "unanswerable",
                "lang": lang,
                "answerable": False,
                "origin": "handwritten",
            }
        )

    OUT.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8"
    )
    print(
        f"{len(rows)} perguntas em {OUT} ({len(CURATED)} aprovadas de {len(candidates)} candidatas)"
    )


if __name__ == "__main__":
    main()
