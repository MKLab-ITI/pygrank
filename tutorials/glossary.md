# Glossary
Term | Explanation
--- | --- 
Seeds | Example nodes that are known to belong to a community.
Graph signal | Scores (not ordinalities) assigned to nodes. They typically assume values in the range \[0,1\].
Graph filter | A strategy of diffusing node scores through graphs through edges.
Personalization | The graph signal inputted in grap filters. This is also known as graph signal priors or personalization vector.
Node ranking algorithm | An algorithm that starts with a graph and personalization and outputs graph signals that boast higher quality than the personalization, for example by making predictions for all graph ndes. Typically consists of combinations of graph filters with postprocessing schemes.

