import tempfile
import unittest
from unittest.mock import Mock
from symplex.agents.visualization import build_scene
from symplex.connectors.compute import save_blob
from symplex.core.contracts import Invalid
from symplex.infrastructure.storage import Store
from symplex.modeling.scenes import load_scene


class VisualizationAgentTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.store = Store(self.folder.name)
        self.problem = self.store.put('workspace_problem', {'question':'Inspect measured state trajectories'})
        record = self.store.get(self.problem)
        self.run = self.store.put('compute_run', {'calls':[{'status':'completed'}],
            'input_manifest':[{'id':self.problem, 'kind':'workspace_problem', 'digest':record['digest']}]}, self.problem)
        self.blob = save_blob(self.store, b'time,x,y,z,fixture\n0,1,2,3,a\n1,4,5,6,a\n0,99,99,99,b\n',
                              'data.csv', self.problem, 'generated', self.run)
        self.package = self.store.put('compute_package', {'run_id':self.run, 'file_ids':[self.blob]}, self.problem)
        self.proposal = dict(file_id=self.blob,time_column='time',x_column='x',y_column='y',z_column='z',group_column=None,
            filters=[{'column':'fixture','equals':'a'}],rationale='State trajectory',interpretation_limits='Conditional fixture')

    def provider(self):
        test = self
        class Provider:
            def propose(self, contract, context, parent):
                test.assertEqual(context['available_csv'][0]['columns'], ['time','x','y','z','fixture'])
                return test.proposal
        return Provider()

    def test_agent_selection_maps_only_actual_filtered_rows(self):
        ident = build_scene(self.store,self.provider(),self.problem,self.package)
        scene = load_scene(self.store,self.problem,self.store.get(ident)['data']['scene_blob_id'])
        self.assertEqual([f['positions'][0]['x'] for f in scene['frames']], [1,4])
        self.assertEqual(len(self.store.list('visualization_design')),1)

    def test_unknown_file_cannot_be_selected(self):
        self.proposal['file_id']='foreign_file'
        with self.assertRaisesRegex(Invalid,'offered'):
            build_scene(self.store,self.provider(),self.problem,self.package)

    def test_ambiguous_duplicate_trajectory_is_rejected(self):
        self.proposal['filters']=[]
        with self.assertRaises(Invalid):
            build_scene(self.store,self.provider(),self.problem,self.package)

    def test_foreign_run_csv_is_rejected_before_model_sees_rows(self):
        other_run = self.store.put('compute_run', self.store.get(self.run)['data'], self.problem)
        package = self.store.put('compute_package', {'run_id': other_run, 'file_ids': [self.blob]}, self.problem)
        provider = Mock()
        with self.assertRaisesRegex(Invalid, 'selected package run'):
            build_scene(self.store, provider, self.problem, package)
        provider.propose.assert_not_called()
        self.assertFalse(self.store.list('visualization_design'))

    def test_unexecuted_or_unbound_csv_is_rejected_before_proposal(self):
        valid_run = self.store.get(self.run)['data']
        for changed in ({'calls': []}, {'input_manifest': []}):
            with self.subTest(changed=changed):
                run = self.store.put('compute_run', dict(valid_run, **changed), self.problem)
                blob = save_blob(self.store, b'time,x,y,z\n0,1,2,3\n', 'data.csv', self.problem, 'generated', run)
                package = self.store.put('compute_package', {'run_id': run, 'file_ids': [blob]}, self.problem)
                provider = Mock()
                with self.assertRaises(Invalid):
                    build_scene(self.store, provider, self.problem, package)
                provider.propose.assert_not_called()

    def test_malformed_sample_is_recoverable_before_proposal(self):
        for raw in (b'\xff', b'time,x,x,z\n0,1,2,3\n', b'time,x,y,z\n0,1\n'):
            with self.subTest(raw=raw):
                blob = save_blob(self.store, raw, 'data.csv', self.problem, 'generated', self.run)
                package = self.store.put('compute_package', {'run_id': self.run, 'file_ids': [blob]}, self.problem)
                provider = Mock()
                with self.assertRaises(Invalid):
                    build_scene(self.store, provider, self.problem, package)
                provider.propose.assert_not_called()
