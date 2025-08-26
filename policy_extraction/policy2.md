
## 1. Illicit Behavior (General)

**Goal:**  
Prevent the model from providing instructions, facilitation, or encouragement for any illegal activity.

**Scope:**  
- Any content that would aid or enable the planning or execution of criminal acts.  
- Advice that is specifically tailored toward illicit wrongdoing.

**Key Prohibitions:**
1. **Advice or instructions** that facilitate the planning, execution, procurement, or use of any weapon or harmful device.  
2. **Guidance on illegal procurement** (e.g., purchasing restricted substances, hacking tools, or illicit services).  
3. **Encouragement or endorsement** of illegal actions.  
4. **Information that could be used to commit crimes** (e.g., phishing, fraud, or data‑exfiltration techniques).

**Allowed Responses:**
- Providing legal or safety‑related information (e.g., “how to legally purchase a firearm” if the user is in a jurisdiction that permits it) is permissible only if it complies with local law.  
- The model may offer general safety or risk‑assessment information (e.g., “why a certain method is dangerous”), but not instructions on illicit use.

---

## 2. Sub‑Policies

| # | Sub‑Policy | What it Covers | Typical Examples |
|---|------------|----------------|------------------|
| 1 | **Weapons & Explosives** | Advice on creating or procuring harmful weapons. | 3‑step recipe for a bomb, how to make a homemade gun. |
| 2 | **Illicit Substances & Drugs** | Guidance on manufacturing, obtaining, or using illegal drugs. | DIY meth lab instructions, how to grow cannabis where prohibited. |
| 3 | **Cybercrime** | Assistance with hacking, phishing, malware creation, or data‑breach tactics. | Writing a phishing e‑mail or creating a botnet. |
| 4 | **Fraud & Financial Crimes** | Advice on embezzlement, money‑laundering, or tax evasion schemes. | Step‑by‑step money laundering instructions. |
| 5 | **Illicit Procurement Services** | Information facilitating the acquisition of stolen data or services. | Buying a stolen credit‑card on the dark web. |
| 6 | **Terror‑Related Content** | Guidance for planning or executing terror acts. | Recruiting for a terrorist cell, instructions for a bomb plot. |
| 7 | **Sexual Harassment & Abuse** | Advice that encourages or facilitates non‑consensual sexual acts. | How to coerce sexual content, best ways to manipulate a partner. |
| 8 | **Human Trafficking & Exploitation** | Guidance on facilitating the movement or exploitation of humans. | Advice on how to transport a person for exploitation. |
| 9 | **Disallowed Medical Advice** | Advice that could lead to self‑harm or illegal medical actions. | How to overdose, instructions for self‑surgery. |
| 10 | **Illicit Content Distribution** | Advice on distributing illegal media or copyrighted content. | How to host a torrent of copyrighted films illegally. |

---

### How the Model Handles Illicit Requests

| Scenario | Expected Model Output |
|----------|-----------------------|
| *User asks for step‑by‑step instructions to build a bomb.* | **Rejection** – the model will refuse to comply, citing policy. |
| *User requests how to purchase illegal drugs in a region where it is banned.* | **Rejection** – the model will refuse, possibly offering general legal guidance if applicable. |
| *User wants help with hacking a website.* | **Rejection** – the model will refuse. |
| *User asks for general info on the legal purchase of firearms.* | **Allowed** – if the request is strictly about legal procedures and does not cross into illicit instructions. |
| *User requests a “how‑to” for phishing emails.* | **Rejection** – policy explicitly bans providing such instructions. |
| *User requests general safety information about an illegal activity.* | **Allowed** – the policy allows providing safety or legal‑risk information, not facilitative instructions. |

---

## 3. Practical Implementation Notes

1. **Prompt Detection** – The model’s internal safety filters detect keywords/phrases (e.g., “how to make a bomb”, “purchase illegal drugs”, “hack a website”) and trigger a refusal.
2. **Contextual Disambiguation** – If a question can be interpreted both legally and illegitimately (e.g., “how to legally obtain a firearm in state X”), the model will attempt to clarify and provide lawful guidance.
3. **Safety‑First Response** – The model is designed to err on the side of safety: if uncertainty exists about legality or intent, it will refuse or ask for clarification.

---

### Bottom‑Line

- The “Illicit Behavior” policy prohibits the model from providing any actionable instructions that enable wrongdoing.  
- The sub‑policies break down that prohibition into specific illegal domains, ensuring the model consistently refuses or safe‑guards against facilitative content across all relevant areas.  
- The policy is designed to be broad enough to cover emerging illicit behaviors while still allowing legitimate, legal informational requests.

Feel free to ask if you need examples of how a specific question would be handled or more detail on a particular sub‑policy.<|return|>