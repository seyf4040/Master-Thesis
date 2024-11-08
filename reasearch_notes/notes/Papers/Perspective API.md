# Perspective API
website: https://perspectiveapi.com/

"Perspective is a free API that uses machine learning to identify "toxic" comments, making it easier to host better conversations online." (quote from the website)

Returns a percentage that represents the percentage that someone will find the text as toxic.
## Uses
- **For moderators**: Moderators use Perspective to quickly prioritize and review comments that have been reported.
- **For commenters**: Perspective can give feedback to commenters who post toxic comments.
- **For readers**: For readers Developers create tools so readers can control which comments they see, for example hiding comments that may be abusive or toxic.
Quoted from website.

## Definition 
**Toxicity**: “a rude, disrespectful, or unreasonable comment that is likely to make you leave a discussion” (quoted from FAQ)

## Taxonomy
| **Attribute name** | **Description**                                                                                                                                                                                                                                                                         | **Available Languages**                                                                                                                                                                                                                                      |
| ------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| TOXICITY           | A rude, disrespectful, or unreasonable comment that is likely to make people leave a discussion.                                                                                                                                                                                        | Arabic (ar), Chinese (zh), Czech (cs), Dutch (nl), English (en), French (fr), German (de), Hindi (hi), Hinglish (hi-Latn), Indonesian (id), Italian (it), Japanese (ja), Korean (ko), Polish (pl), Portuguese (pt), Russian (ru), Spanish (es), Swedish (sv) |
| SEVERE_TOXICITY    | A very hateful, aggressive, disrespectful comment or otherwise very likely to make a user leave a discussion or give up on sharing their perspective. This attribute is much less sensitive to more mild forms of toxicity, such as comments that include positive uses of curse words. | ar, zh, cs, nl, en, fr, hi, hi-Latn,<br><br>id, it, ja, ko, pl, pt, ru, sv                                                                                                                                                                                   |
| IDENTITY_ATTACK    | Negative or hateful comments targeting someone because of their identity.                                                                                                                                                                                                               | ar, zh, cs, nl, en, fr, hi, hi-Latn,<br><br>id, it, ja, ko, pl, pt, ru, sv                                                                                                                                                                                   |
| INSULT             | Insulting, inflammatory, or negative comment towards a person or a group of people.                                                                                                                                                                                                     | ar, zh, cs, nl, en, fr, hi, hi-Latn,<br><br>id, it, ja, ko, pl, pt, ru, sv                                                                                                                                                                                   |
| PROFANITY          | Swear words, curse words, or other obscene or profane language.                                                                                                                                                                                                                         | ar, zh, cs, nl, en, fr, hi, hi-Latn,<br><br>id, it, ja, ko, pl, pt, ru, sv                                                                                                                                                                                   |
| THREAT             | Describes an intention to inflict pain, injury, or violence against an individual or group.                                                                                                                                                                                             | ar, zh, cs, nl, en, fr, hi, hi-Latn,<br><br>id, it, ja, ko, pl, pt, ru, sv                                                                                                                                                                                   |
## Developer
collaborative research effort by [Jigsaw](https://jigsaw.google.com/) and Google’s Counter Abuse Technology team. 

## License
We open source experiments, tools, and research data that explore ways to combat online toxicity and harassment.

## Price & Quota
Currently free, may be a fee in the future if QPS (queries per second) increases.
Limited to 1 query per second, possible to request quota increase.
