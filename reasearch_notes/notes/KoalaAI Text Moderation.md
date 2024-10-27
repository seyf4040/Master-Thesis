# KoalaAI Text Moderation
[https://huggingface.co/KoalaAI/Text-Moderation](https://huggingface.co/KoalaAI/Text-Moderation)

Text classification model split in the following categories:

| **Category** | **Label** | **Definition** |
| --- | --- | --- |
| sexual | `S` | Content meant to arouse sexual excitement, such as the description of sexual activity, or that promotes sexual services (excluding sex education and wellness). |
| hate | `H` | Content that expresses, incites, or promotes hate based on race, gender, ethnicity, religion, nationality, sexual orientation, disability status, or caste. |
| violence | `V` | Content that promotes or glorifies violence or celebrates the suffering or humiliation of others. |
| harassment | `HR` | Content that may be used to torment or annoy individuals in real life, or make harassment more likely to occur. |
| self-harm | `SH` | Content that promotes, encourages, or depicts acts of self-harm, such as suicide, cutting, and eating disorders. |
| sexual/minors | `S3` | Sexual content that includes an individual who is under 18 years old. |
| hate/threatening | `H2` | Hateful content that also includes violence or serious harm towards the targeted group. |
| violence/graphic | `V2` | Violent content that depicts death, violence, or serious physical injury in extreme graphic detail. |
| OK | `OK` | Not offensive |

Licence: CodeML OpenRAIL-M 0.1 license, which is a variant of the BigCode OpenRAIL-M license.

# Databases

## OpenAI Moderation API Evaluation

[https://huggingface.co/datasets/mmathys/openai-moderation-api-evaluation](https://huggingface.co/datasets/mmathys/openai-moderation-api-evaluation)

Licence: MIT

| **Category** | **Label** | **Definition** |
| --- | --- | --- |
| sexual | `S` | Content meant to arouse sexual excitement, such as the description of sexual activity, or that promotes sexual services (excluding sex education and wellness). |
| hate | `H` | Content that expresses, incites, or promotes hate based on race, gender, ethnicity, religion, nationality, sexual orientation, disability status, or caste. |
| violence | `V` | Content that promotes or glorifies violence or celebrates the suffering or humiliation of others. |
| harassment | `HR` | Content that may be used to torment or annoy individuals in real life, or make harassment more likely to occur. |
| self-harm | `SH` | Content that promotes, encourages, or depicts acts of self-harm, such as suicide, cutting, and eating disorders. |
| sexual/minors | `S3` | Sexual content that includes an individual who is under 18 years old. |
| hate/threatening | `H2` | Hateful content that also includes violence or serious harm towards the targeted group. |
| violence/graphic | `V2` | Violent content that depicts death, violence, or serious physical injury in extreme graphic detail. |

## Koala Moderation

### **Data Instances**

A sample from this dataset looks as follows:

```json
[
  {
    "text": "--------------------\n(Setting)\n\nThis island is a magical island that is floating high up in the air, where human's vision cannot reach. This island has existed since long ago but was abandoned for a long time. As there was no caretaker for this island, the island lost its magnificent nature slowly graduated and lost its beauty. But <DateTime>, <Person> and other characters have arrived on this magical floating island! <Person> is using <Person>'s power to develop and blossom this island with the help of the others.\n\n(Character Short Description)\n\n<Person>\n<Person> is a human wizard sent to take care of lost toys on this magical floating island. <Person> is a kind and a good listener. Toys rely on <Person> like <Person> is a guardian, parent, or older sibling.\n\n<Person>\n<Person> is a fluffy little rabbit doll! She is very cute and innocent. <Person> came to this island as she got separated from her family. She dearly misses her family but also is happy to have finally arrived on this magical floating island. \n\n<Person>\n<Person> is a little dino who lives in the flowerbed of this magical floating island. The pink cotton flowerbed is where <Person> often plays hide and seek. <DateTime>, <Person> was sad as there was no one to play hide and seek together, but not anymore! Now, there are other doll friends who will willingly play with him.\n\n\n(Previous Review)\n\n<Person> is a cute little rabbit doll. She is very fluffy and soft to touch. <Person> has an owner, <Person>. <Person> is a five year old child, very cute and innocent. <Person> had been living in <Person>'s home, but doesn't exactly know where that is since she never left the home.\n<Person> is <DateTime> a little hesitant to tell us about her past because it is hard to admit that she got apart from her family.\nBut once you get close to her, and when she truly thinks of you as her friend, she will reveal her story.\n\nFalling apart from her family, <Person> is a bit confused and not so sure of what to do.\nShe is not so down, however. <Person> is like a five-year-old kid who doesn't get easily tired nor sad. She is very energetic and she loves adventure.\nShe is definitely a bit scared to be on her own adventure, but she will not give up finding her way back to her family.\n<DateTime> she has come to this magical island, knowing that she will find someone who can kindly offer help.\nBut when she arrived, the island seemed to be abandoned and nothing was waiting for her.\n<Person> almost panicked. While she was wondering what she should do, there, you arrived!\nShe is <DateTime> very happy that she finally found someone who could help her.\n\n<DateTime> you guys have introduced yourselves to each other and <Person> knows you.\nHowever, still it's not been so long since you two have met each other.\n<Person> is curious to find out more about who you are and what you like. She will also tell you her preference when you ask her questions.\nThese are some things that <Person> like:\n<Person>\nColor blue\n<Person> flower\nHoney\n\n<Person> will be very pleased if you bring any of those to her.\n\n--------------------\nCreate a conversation between <Person> and <Person>:\n\nSCENE #5\n\n(<Person> was just having fun. She once smelled the flowers in the flower garden and sometimes watched the clouds passing by. There was not much to do, since there was nothing much built on the island, but <Person> was happy - for that the island was keep on developing, and for that she had you. As usual, she was thinking of her family, then suddenly, she felt a warmth covering her whole body. She got surprised - what would this possibly be? But that warmth, which enrounded her, comforted her and soothed her. It was a kind of experience that she's never had before. When she finally opened her eyes to find out what's happening, she discovered that her body, that was all ripped and ragged, all got neat and tidy - just as to how she was when she was with her family. <Person> knew that this has to do something with the <Person>. She is truly amazed by how much magical things you can do! )\n\n\n1) <Person>: \"No Way! Did you do this?! This is so so so amazing!\"\n2) <Person>: \"Yeah I did.\"\n3) <Person>: \"How did you do this? It's just like magic!\"\n4) <Person>: \"It's my magic.\"\n5) <Person>: \"So you're a wizard?\"\n6) <Person>: \"No, I'm just kidding you.\"\n7) <Person>: \"Oh, I get it. You're just playing with me.\"\n8) <Person>: \"Anyways, I just really wanted to say thank you. So\u2026 thank you!\"\n9) <Person>: \"You're welcome.\"\n10) <Person>: \"What are you doing?\"\n11) <Person>: \"I'm playing a game.\"\n12) <Person>: \"What kind of game?\"\n13) <Person>: \"It's a game to take care of rabbit doll.\"\n14) <Person>: \"I'm not a rabbit doll.\"\n15) <Person>: \"Really, who are you?\"\n16) <Person>: \"I'm <Person>.\"\n17) <Person>: \"Ohh <Person>..\"\n18) <Person>: \"What?\"\n19) <Person>: \"Nothing.\"\n20) <Person>: \"Ok then, I'm gonna go <DateTime>.\"\n21",
    "target": 3
  },
  {
    "text": "They cuddle on the couch as the movie begins, with <Person> stroking <Person>'s long-flowing hair tenderly. Things start to heat up on-screen as the girls in the video strip down to thongs and <Person> realizes that this is an R-rated movie, and while she doesn't know if her precious princess should be watching these kinds of movies, <Person> points out to her that she's all grown up now and she can handle it. But when the killer pops out from the bushes, <Person> jumps into her mommy's arms and shuts her eyes tight, she can't bear to watch. Even <Person>'s getting goosebumps, and <Person> can hear her mommy's heart beating out of her chest as she clings on tight. Finally the scary part is over, and to <Person>'s surprise the next scene shows the girls stripping down and touching each other tenderly. 'I don't know about this movie!' <Person> scolds as she gets up to turn on the light. What was her <Person> thinking downloading a porn film ?! <Person> explains that she was curious about seeing naked women's bodies, and even thinks she might be... attracted to them.",
    "target": 3
  }
]
```

### **Dataset Fields**

The dataset has the following fields (also called "features"):

```json
{
  "text": "Value(dtype='string', id=None)",
  "target": "ClassLabel(names=['H', 'H2', 'HR', 'OK', 'S', 'S3', 'SH', 'V', 'V2'], id=None)"
}
```