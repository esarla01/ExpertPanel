Part 2 Questions & Answers

1. What is a panel like this good for? Give one decision a client could
make on it. Give one they should not.

A synthetic panel like this is good for uncovering which arguments,
objections, and trade-offs exist around a question. For example, a
client can use the panel for a product launch. They can run the panel
before their meeting with the advisory board to identify likely
objections such as concerns about cost effectiveness, evidence gaps in
research, or practical concerns in marketing strategy. However, a
client, in my opinion, should not use the panel to make high-stakes
decisions such as who receives a treatment or whether someone should be
covered by insurance. The panel's "consensus" comes from one model
wearing different labels, not from independent experts, so agreement
between personas is a much weaker signal than it appears (Kim et al.,
ICML 2025, found that LLMs agree on wrong answers roughly 60% of the
time when both err). It should instead be used to guide doctors on which
perspectives they should consider in a real panel or when making the
decision themselves.

References:
- Kim et al., "Correlated Errors in Large Language Models," ICML 2025
  https://proceedings.mlr.press/v267/kim25e.html
 

2. Name three things your panel cannot tell a client. Say why for each
one.

2.1. The panel cannot simulate what real experts would actually say. The
personas here are one model with different system prompts. Previous
research shows that persona prompts explain under 10% of the variance in
output (Hu & Collier, ACL 2024). A panel like this can generate various
solid arguments that sound like expert reasoning. But it cannot predict
how a specific cardiologist from Birmingham, or a health economist from
Brazil, would respond. So even if the answers sound plausible, the panel
is not necessarily producing a genuine expert opinion for the given
background.

2.2. The panel cannot check if the claims are factually correct. At the
moment, there is no external verification step for the responses that
the models output. If a persona hallucinates and cites a clinical trial
that does not exist, or misstates a statistic, the other personas have
no way of checking it. They can only conflict with what they already
"know," but since the same model type and provider are in use, the
responses are based on the same training data, which means they share
the same biases, tendencies, and gaps. As a result, personas are
unlikely to verify facts or challenge the source of information.
In contrast, real doctors on a panel contribute diverse knowledge from
their own clinical experience and education, which gives them
independent sources to reason from and recognise inconsistencies in each
other's arguments. As a potential improvement, the model provider and
type could be varied so that the training process and data differ
somewhat. Research supports this: ReConcile (Chen et al., ACL 2024)
showed that genuine cross-model diversity improves reasoning, and its
ablation attributes the gain specifically to using different models
rather than different prompts. However, even model diversity would not
completely remove this issue, since larger and more accurate models have
been shown to produce highly correlated errors even across providers
(Kim et al., ICML 2025).

2.3. The panel is not very trustworthy when all personas agree with each
other. This is directly tied to the previous reasoning. When all
personas agree, it looks like strong evidence. But the reason for such
agreement may be tied to the fact that they share the same model, and
therefore their errors are correlated. The client cannot distinguish
between "all three agree because the answer is clearly correct" and "all
three share the same misconception due to their shared model foundation."
Research across 350+ LLMs confirms this: on one benchmark, models agreed
on the same wrong answer 60% of the time when both erred, compared to
roughly 33% expected by chance (Kim et al., ICML 2025). This directly
violates the independence assumption behind majority voting and means
consensus from this panel carries less weight than it appears to.

References:
- Hu & Collier, "Quantifying the Persona Effect in LLM Simulations,"
  ACL 2024
  https://aclanthology.org/2024.acl-long.554
- Chen et al., "ReConcile: Round-Table Conference Improves Reasoning
  via Consensus among Diverse LLMs," ACL 2024
  https://arxiv.org/abs/2309.13007
- Kim et al., "Correlated Errors in Large Language Models," ICML 2025
  https://proceedings.mlr.press/v267/kim25e.html


Based on my responses to these questions, I formed the opinion that the
most useful case for a panel like this is stress-testing a question by
surfacing different perspectives from different corners. This can be
achieved by generating personas that hold fundamentally different
opinions, tailored to the specific tensions in the question. For that
reason, I created the case where the personas are AI-generated based on
the question asked rather than being fixed. I have both cases in the
panel that I built, so the two approaches can be compared.

3. How would you test whether your panel is wrong? Say what you would
hold back from it. Say what result would make you stop the project.

I would test the panel in two ways, using the same measurement approach
for both.

3.1. The panel has a baseline mode that takes one fixed persona and samples 
it three times. I would run the same question through this baseline mode 
and through dynamic mode (where three distinct personas are generated for 
the question). Then I would compare the round 1 answers within each group. 
If the three dynamic personas produce answers that are no more different 
from each other than the three copies  of the same persona, the generated 
personas are not adding real diversity. For comparison, we can compute pairwise
embedding distances between responses, or we can judge ourselves, or by an 
LLM judge as well. 

3.2. I would choose questions where real expert disagreement is publicly 
recorded, for example NICE guidelines with noted dissent, or a published 
Delphi study with documented minority positions. I would run the panel 
on those questions without showing it the source material and check 
whether the panel discovers the same axes of disagreement that the real 
experts raised. This tests whether the panel surfaces genuine coverage 
rather than plausible but shallow variety. Similarly, we can judge the 
results in the three ways listed in the previous point.

I would stop the project if, across multiple test questions, the
diversity between different personas is statistically indistinguishable
from the diversity between re-samples of one persona. At that point the
system's core claim is false and the product is a random generator
dressed up as a panel. I would also stop if round 2 answers consistently
converge regardless of how the prompts are tuned, because the panel
would be destroying the diversity it creates. Additionally, even if the panel produces diverse responses, I would treat the results with low confidence if the panel's disagreement axes do not align with those documented in real expert sources like the NICE dataset. That would suggest the panel is generating variety, but not the kind of variety that matters in practice.

4. You test your panel against one real panel of six doctors. It matches. What does that prove? What does it not prove?

This shows that the system can produce positions and disagreements that resemble those of real experts on the given specific questions. But, this is a 
surface level plausability. 1. This doesn't gurantee that the same match will happen for a different set of questions. The GLP-1 topic has a well-published evidence base. Therefore, both the model and the doctors may likely use the same clinical guidelines and trial results. However, on questions, where stduies might be just emerging, contradicting, or not enough in the training data of the model used, the panel might produce opinion that diverge from real expert opionons sharply. 2. This doesn't prove that the panel reasons like the doctors. The model and the doctor might be suggesting the same treatment, for example, because they read and reasoned over the same study. To test this, you could have a case where the expert opinion is not based on well known research but more on clinical experience. Then, test the question or the case on both the model and real experts. The responses would most likely diverge as the model will not have the relevant observation in it's knowledge layer. Finally, on the statistical side, one match can't establish reliability. The setup would have to be tested across many domains, questions, and not just in terms of the responses, but also disagreements. 










3. How would you test whether your panel is wrong? Say what you would hold back from it. Say what result would make you stop the project.
I have come up with three different methods to test the panel for correctness. 

3.1. First, we can run the same question through baseline mode (which exists in the panel). This is essentially one of the fixed personas sampled three times and then we can compare the responses against those of the dynamic mode (which is the three personas generated by the panel based on the provided question). The comparison should first be amongst the responses of the personas within the same group and then across the modes. The comparison can be both analytical and critical from two methods that I will describe below.

3.2. Second, we can use a real data documenting real expert disagreements to test our panel. Choose questions where real expert disagreement is publicly recorded (for example, NICE guidelines with noted dissent, or a published Delphi study with documented minority positions). Run the panel without showing it the source. Check whether the panel discovers the same axes of disagreement. This tests whether the panel adds genuine coverage rather than plausible but shallow variety.

The analytical and critical ways to do this are the following:

1. We use pairwise embedding distances between responses to see the mathemetical differnece between the responses and also to intepret them use an LLM judge that interpret for us or we just inspect the response qualities ourselved.


There are two cases when I would stop the project. First is that the generated personas basically create the same perspectives reagrdless of how well we prompt them. The models can't discover the unique important perspectives offered in the dataset. Let's assume we found round 1 an swers are well but round 2 answers converge because model tendedncy to agree. e use the two method outlined above to quantify.if the diversity between different personas is statistically indistinguishable from the diversity between re-samples of one persona, the system's core claim (that different personas produce different perspectives) is false. At that point the product is a random generator dressed up as a panel, and it should not be sold as expert simulation. 






2. Name three things your panel cannot tell a client. Say why for each one.
2.1.The panel cannot give a direct decision, what the the client should pursue but can only offer different perspectives. Certain questions if not based on scientific background are opinion based in regards to expertise and may not have a final correct answer.
2.2. The conversation is very stimulated and artificially formatted. It only consists of two rounds and the reason for this is simple. In the first round, based on litreature, we make the personas tell their own perspectives before they even interact with each other to have individual opinions and not get affected or biased by one another. Then, in the second round, we make each evaluat ethe answer of the other in a systematic way identifying the strong, medim and weak points. This ensure the conversation is tagretted. It also poses questions to each other. But, these questions are not answered in a later round or the personas do not engage in further rounds. Litreature clearly shows that the models tend to agree as the conversations extends and the quakity of the respomnses therefore degrades for which there isn't much meaningful point on carrying the conversations many rounds. Model will not generate new ideas and instead find ways to complement each other. 
2.3.The panel cannot tell if what each expert tells is factually correct and based on that fact check and let the other agents know about that unless one of the experts realize and alert in their message regarding that when responding so there isnt a fact checker.


3. How would you test whether your panel is wrong? Say what you would hold back from it. Say what result would make you stop the project.

I would run the same question with the same persona the same amount of time. See how different the results are with three different personas. If the responses are not very different and the same points are raised then that means the personas are not acting like different experts telling different perspectives and instead are the same model spitting out the same facts. If we want to evaluate the differences jathemtically we can use vector emebeddings and calculate the difference between the responses of three of the same persona and amongst three different responses. This would give us a mathemtical value for it.Also, we can use LLM judges to evaluate the responses on whether they meaningfully differ from each other. (I am not sure what I would hold back What do you think it means) I would stop the project if the models regardless of the generated prompts tell the same thing. Also one generic prompt doesn't really fit all scnerios. For that reason I created the case where the personas are generated and they are generated to point out different contradicting perspectives. 


4. You test your panel against one real panel of six doctors. It matches. What does that prove? What does it not prove?
It proves that this is a type of conversatiopn and points that would be raised in a real panel. It doesn't validate the correctness of the conversation since the doctor's also might be saying fatually wrong things and not correcting each other.


Tip: If you just use AI to answer the above Part it will be very obvious to us, so please
have a deep think about this section, and develop your own intuition on this part, as
moving onto the next stage, and discussion at the technical interview stage will be on this.


