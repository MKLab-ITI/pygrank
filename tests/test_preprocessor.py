
def test_preprocessor(self):
    from pygrank import preprocessor, MethodHasher
    G = test_graph()
    with pytest.raises(Exception):
        pre = preprocessor(normalization="unknown", assume_immutability=True)
        pre(G)

    pre = preprocessor(normalization="col", assume_immutability=False)
    G = test_graph()
    res1 = pre(G)
    res2 = pre(G)
    self.assertTrue(id(res1) != id(res2),
                    msg="When immutability is not assumed, different objects are returned")

    pre = MethodHasher(preprocessor, assume_immutability=True)
    G = test_graph()
    res1 = pre(G)
    pre.assume_immutability = False  # have the ability to switch immutability off midway
    res2 = pre(G)
    self.assertTrue(id(res1) != id(res2),
                    msg="When immutability is not assumed, different objects are returned")

    pre = preprocessor(normalization="col", assume_immutability=True)
    G = test_graph()
    res1 = pre(G)
    pre.clear_hashed()
    res2 = pre(G)
    self.assertTrue(id(res1) != id(res2),
                    msg="When immutability is assumed but data cleared, different objects are returned")