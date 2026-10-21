import unittest
from experiments.campaign_numerical_external_forces import set_forcing_flag


class FlagTests(unittest.TestCase):
    def test_only_documented_boolean_flag_is_authored_and_checked(self):
        class Attribute:
            def Set(self,v):self.value=v;return True
            def Get(self):return self.value
            def HasAuthoredValueOpinion(self):return True
        class Prim:
            def __init__(self):self.attr=Attribute()
            def GetAttribute(self,name):
                self.name=name;return self.attr
        class Stage:
            def __init__(self):self.prim=Prim()
            def GetPrimAtPath(self,path):
                self.path=path;return self.prim
        stage=Stage();record=set_forcing_flag(stage,False)
        self.assertFalse(record['value']);self.assertEqual(stage.path,'/physicsScene')
        self.assertEqual(stage.prim.name,'physxScene:enableExternalForcesEveryIteration')
        self.assertTrue(set_forcing_flag(stage,True)['value'])
        with self.assertRaises(ValueError):set_forcing_flag(stage,'false')


if __name__=='__main__':unittest.main()
