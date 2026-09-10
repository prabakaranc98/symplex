import unittest
from symplex.agents.planner_context import bounded_planner_context
from symplex.core.contracts import canonical, Invalid

class PlannerContextTests(unittest.TestCase):
    def test_history_cannot_crowd_out_current_intent(self):
        source={'problem':{'question':'Estimate the effect under the revised expert constraints'},
                'instruction':'trusted instructions','budget':{'usd':1},
                'completed_actions':[{'tool':'represent_system','artifact_id':str(i),'result':{'system_id':str(i),'representation':'x'*9000}} for i in range(24)],
                'working_context':[{'id':'s','digest':'abc','kind':'complex_system','data':{'body':'界'*20000}}]}
        packed=bounded_planner_context(source)
        self.assertLessEqual(len(canonical(packed).encode()),48000)
        self.assertEqual(packed['problem'],source['problem'])
        self.assertEqual(packed['budget'],source['budget'])
        self.assertTrue(packed['working_context'][0]['data']['truncated'])
        self.assertEqual(packed['context_selection']['history_total'],24)
        self.assertEqual(source['working_context'][0]['data']['body'],'界'*20000)

    def test_rejections_and_artifact_references_survive_summarization(self):
        value=bounded_planner_context({'completed_actions':[{'tool':'compare_computation','artifact_id':'a','result':{'status':'rejected','error':'Frozen protocol mismatch','comparison_id':'b'}}]})
        self.assertEqual(value['completed_actions'][0]['result']['error'],'Frozen protocol mismatch')
        self.assertEqual(value['completed_actions'][0]['result']['comparison_id'],'b')

    def test_oversized_current_direction_fails_instead_of_silent_loss(self):
        with self.assertRaises(Invalid):
            bounded_planner_context({'instruction':'x'*50000})

    def test_newest_tool_observation_survives_before_older_working_memory(self):
        result = {'hits':[{'text':'critical new observation'}]}
        context = {'instruction':'trusted', 'completed_actions':[
            {'tool':'search_artifacts','artifact_id':'a','result':result}],
            'working_context':[{'id':str(i),'digest':'d','data':{'body':'x'*950}} for i in range(20)]}
        packed = bounded_planner_context(context, max_bytes=2000)
        self.assertEqual(packed['completed_actions'][-1]['result']['observed_result'], result)
        self.assertTrue(packed['context_selection']['omitted_artifact_ids'])
