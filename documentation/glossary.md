# Glossary
More detailed descriptions of the following terms can be found in the [documentation](documentation.md).

Term | Explanation
--- | --- 
Graph neural network | A combination of traditional neural networks and node ranking algorithms.
Graph signal | Scores assigned to nodes. Each node's score typically assumes values in the range \[0,1\].
Graph filter | A strategy of diffusing node scores through graphs through edges.
Node ranking algorithm | An algorithm that starts with a graph and personalization and outputs graph signals that boast higher quality than the personalization, for example by making predictions for all graph ndes. Typically consists of combinations of graph filters with postprocessing schemes.
Personalization | The graph signal inputted in grap filters. This is also known as graph signal priors or the personalization vector.
Seeds | Example nodes that are known to belong to a community.
Tuning | A process of determining algorithm hyper-parameters that optimize some evaluation measure.
