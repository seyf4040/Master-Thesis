# Design and Application of an AI‐Based Text Content Moderation System
Site web: https://onlinelibrary.wiley.com/doi/full/10.1155/2022/2576535

# Introduction
Text content moderation (TCM), for online educational platform. Keyword matching moderation ignores context and thus raises a lot of false positives.

## Architecture design of the AI-base TCM system
AI moderation + manual recheck. Developed using AI cloud service platform.
Input text is first analysed by frontend for formatting then api request is made for moderation.
Different analysis performed: 
- text analysis
- lexical analysis
- syntax analysis
- semantic analysis
- sentiment analysis
- text classification
Taxonomy: 
 - pornographic content
 - terrorism content
 - advertising content
 - illegal content
 - abusive content
When model is unable to classify sample, the sample is pushed to system administrator for manual moderation. 
classified content will go to database so the model can perform updates and self-learn (Active learning).
Algorithm used: FastText.

## Dataset 
contains about 360k samples. available upon request to authors.

## Experiments 
Too light, need bigger test/validation dataset.

## ChatGPT's review
# Detailed Review of "Design and Application of an AI-Based Text Content Moderation System"

## Summary
The paper explores the development and deployment of an AI-based Text Content Moderation (TCM) system. The system leverages cloud computing and machine learning models to moderate text in real-time, particularly for an educational platform. It combines rule-based and AI-driven techniques to improve the accuracy and efficiency of content filtering, reducing reliance on manual moderation. The authors provide a technical breakdown of the system's architecture, including its modular layers, reliance on Tencent Cloud infrastructure, and implementation of the FastText model for text classification.

## Strengths
- **Relevance and Application**: The paper addresses a pressing need for effective moderation in online educational platforms, focusing on reducing harmful or disruptive content.
- **Technical Detail**: A thorough explanation of the system’s architecture and modular breakdown (hardware, middleware, algorithm service, and application service layers) supports a clear understanding of its functionality.
- **Efficient Use of AI**: The integration of FastText and cloud-based resources demonstrates the system’s ability to handle high text volumes with good accuracy (98.3% in testing).
- **Cloud-Based Scalability**: The use of Tencent Cloud services enables flexible scaling to meet various computational demands, suggesting that the system can handle increased workloads.
- **Continuous Learning and Adaptation**: The authors implemented mechanisms for the model to update its sample library based on flagged or moderated content, which will improve its adaptability to emerging patterns in harmful content.

## Weaknesses
- **Bias in Text Classification**: Although FastText performs well with Chinese text, the study does not address bias in handling nuanced or sensitive content, which can lead to inappropriate moderation.
- **Lack of Comparative Analysis**: There is no comparative analysis with other TCM systems, such as Google’s Perspective API, to benchmark the model’s performance across various content moderation needs.
- **Limited Contextualization of Results**: While the accuracy rate is noted, the paper lacks a breakdown of performance across content types (e.g., spam, harassment, misinformation) or false positive/negative rates for each.
- **User Experience Consideration**: The paper does not address user feedback or experience after moderation interventions, which is vital in educational contexts where censorship can affect open dialogue.
- **Limited Discussion of Multilingual Support**: Since the system operates primarily in Chinese, its broader applicability to multilingual educational environments is unclear.

## Minor Comments
- **Figures and Diagrams**: Figures (e.g., Figures 1 and 2) detailing the system’s architecture would benefit from more descriptive captions to assist readers in following complex processes.
- **Terminology Consistency**: Terms like “AI-based moderation” and “intelligent moderation” are used interchangeably but could be defined to clarify specific methods or approaches within the system.
- **Future Directions**: While the conclusion mentions the potential of the TCM system in education, a more detailed roadmap for future improvements or upgrades would strengthen the paper’s relevance.

## Recommendation
The paper presents a well-designed and technically sound AI-based content moderation system. It offers practical applications and a strong use of cloud-based resources, making it suitable for educational settings. However, the paper requires some minor revisions, specifically to address bias issues, expand on multilingual capabilities, and provide a comparative analysis. My recommendation is **acceptance with minor revisions** to enhance the study’s applicability and depth.