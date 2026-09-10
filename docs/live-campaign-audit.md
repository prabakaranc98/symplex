# Live campaign audit

Updated: 2026-09-10T20:27:34+00:00. This is a point-in-time artifact audit while scoped campaign retries may still be active.

The four-case campaign exercises the investigation software. It has not established scientific, ecological, clinical or future-election predictive validity. A succeeded job or a source-linked narrative is not evidence of executed numerical work. Counts below distinguish records since each case’s latest scoped-attempt record from earlier records; subsequent probes in the same case scope are included and are not attributed to another concurrent case.

## Budget and observation method

- Shared campaign-workspace ledger: $35.798820 used or reserved against its persisted $40.00 ceiling. The original four cases each retain a $10 scope ceiling; additional Case I has a $5 scope ceiling, including earlier attempts.
- Model requests used or reserved: 167; evidence requests: 29. 12 reservations are unsettled. These are local accounting estimates/reservations, not a provider invoice.
- Scope totals, not overlapping global attempt spans, attribute concurrent work. Global totals include all cases sharing this workspace ledger, including any additional cases; the per-case sections cover the original four and additional Case I. The other workspace and unobserved external charges are outside this document.
- SQLite was read with mode=ro in a read transaction; immutable artifact JSON and the per-case /private/tmp/symplex-*-v*.jsonl logs were inspected. No live Store initialization, runtime changes or paid calls were made by this audit.

## Current findings

- Protein has one saved protocol marked frozen_for_comparison, but its own comparison_controls explicitly states that conservation and positivity checks are incomplete and says not to execute. The stored state and that declared readiness limitation conflict. No compute run followed: [experiment_protocol_fd4ff883830e4411](../.symplex-campaign/artifacts/632ad07feff9a62288c71e4d55a626848b5e7dd4f58cbd9e19f603a20cf2c0f8.json).
- Protein and additional Case I each produced a requires_review decision brief. Both defer empirical or deployment conclusions; the latter describes an unexecuted synthetic router comparison. Neither is evidence of measured performance: [decision_brief_bbd35b965126403c](../.symplex-campaign/artifacts/010f4c4a1747610fea9e0f55f8c396e583c58dd2e24d3c644e0c1459c3e3ce83.json), [decision_brief_84a88b30979849d7](../.symplex-campaign/artifacts/a8187e5a8dbe458a83d2703785838fccde47ba46cc9ffd735acb2b811dda2902.json).
- Actual OpenAI Agents SDK 0.20.0 execution returned seven native Responses for Hudson with one submit_proposal invocation each. The two ExperimentProtocol responses reached the host, which rejected a time/value-column collision, then empty comparison controls, uncertainty plan and stopping rule. SDK execution succeeded; protocol acceptance did not: [sdk_agent_run_e9f0746b27d94dbb](../.symplex-campaign/artifacts/4ddc2b56dc09c49cc92709fa054e1e889c2cd1a29f29828745ce9549b441fe33.json), [sdk_agent_run_e44e48821e244c98](../.symplex-campaign/artifacts/f19d28c80ddbc8ac0da0a498f7804ba91950f2b79a0b923c5363115af3dcd7ae.json), [proposal_rejection_1fb906974fa54cac](../.symplex-campaign/artifacts/4b33ebbd49af05c2859588d135c30be3118c26afb2442122579d753556354593.json), [proposal_rejection_7254ffc04f2e4254](../.symplex-campaign/artifacts/5692732233dc05477cc1d9f951e115c05d268e59c60eedd1fbe8a3d2ffb87eaa.json).
- The subsequent Hudson SDK investigation ended budget-exhausted before its next protocol request. Scope usage remains below its nominal ceiling because the conservative next-request reservation would not fit; this is admission control, not free remaining capacity sufficient for that request: [sdk_agent_run_1f461c7b67214915](../.symplex-campaign/artifacts/2c0894cf5c6248694d3768857d447a32ebd2a4e90064e549e6d958ea09348bcc.json), [failure_9cf15353e5074547](../.symplex-campaign/artifacts/b3fef486be8e4df00dc44355f621db11945fb39ffb82e751fa50307621a2df07.json).
- Initial SDK probes for data-center and Hudson failed with APIConnectionError in under 0.3 seconds. No completed native response was recorded for those probes, and their reservations were retained. These failures establish neither a scientific result nor a defect in the generated proposals: [model_failure_c04cae0236294c1b](../.symplex-campaign/artifacts/139e89a23a07745035e20e98933e78154a0a10da72c05a726ed664e227afae2d.json), [model_failure_ae759fa4ade145a6](../.symplex-campaign/artifacts/4b5e29b3fe1f71e97673cd8e7201d9b98a7e4d09412cd8e5b5e3fdec4540473a.json).
- The separate synthetic biology interface demonstration and its generated narration belong to the original workspace, not these live campaign cases. They must not be presented as completion of this campaign or evidence of general recursive improvement.
- The v4-and-later cohort currently has 0 compute runs, 0 packages, 0 host numerical verifications, 0 comparisons and 1 decision briefs. Presence alone does not establish that any checks passed.
- The v4-and-later cohort has 0 raw-source provenance records and 0 external-source blobs. Research notes and citation links are tracked separately from downloaded data bytes.
- Prior data-center and protein jobs ended with dependency_gap deliverables despite succeeded job status. Neither old deliverable supplies a compute package, comparison, model_critique or decision_brief.
- Recorded older blockers include incomplete provider responses in data-center representation; a provider timeout and a rejected port/state time-scale relationship in protein representation. These are implementation/runtime or contract failures, not adverse domain evidence.
- Prompt versions and tool descriptions are preserved in run manifests. planner_input_ack records identify the input revision presented at a planning boundary and explicitly do not prove incorporation. Model-call request digests do not, by themselves, expose the complete actual request/context payload.
- Data-center, Hudson and protein v4 all obtained typed representations but downstream review/planning encountered context-envelope rejection. These accepted representations are proposed models; none by itself constitutes a frozen experiment or execution.
- Data-center HITL steering was presented at the next recorded planning boundary; the chosen action explicitly classified it as presentation/uncertainty guidance that preserved Problem DNA. This verifies a recorded response for one integration probe, not general instruction-following quality.
- Protein v4 obtained a typed representation but review_hypotheses and plan_experiment were both rejected immediately with Context exceeds the bounded investigation envelope. The attempt then delivered dependency_gap without a protocol or computation. This is a host context-assembly failure, not an empirical domain result.
- A concrete retrieval limitation occurred: search_artifacts indexes context/evidence_note/solution/model_critique only and returns top-five lexical chunks. Data-center requested a complete saved solution twice, and protein requested its new complex_system, but responses could not provide complete requested contracts. This is a tool/context access mismatch, not evidence that those artifacts do not exist.
- The first post-v4 retries of data-center, Hudson and protein all failed after selecting inspect_artifact with limits of 18000/24000 against an 8000 cap. Typed-input validation escaped into whole-job failure. No artifact read occurred in those attempts; this is a host tool-contract/handling defect.
- Runtime versions can differ across retries. v4 prompt snapshots predate the later model-map requirement. A missing model map in a v4 package must be recorded as unavailable traceability, not a domain-model failure; symbol/column binding is also not semantic or scientific validation.
- The generic public-file tool is available in current manifests. Its operator-owned exact-host policy may reject a newly discovered source outside the catalog allowlist; availability must not be described as successful raw-data acquisition.

## data_center

- Problem: [workspace_problem_b0646cba22024069](../.symplex-campaign/artifacts/21876e2509311bb48595c65093d0ab1f2babda478c3b1b09855d7f10d88f0f2b.json); latest attempt: [campaign_attempt_434f5938b74847ee](../.symplex-campaign/artifacts/fc374455631232f335bac4cea76c4d39ca377346f06539e1e2648b862f94d376.json). Latest recorded job status since that attempt began: succeeded; attempt end: [campaign_attempt_end_654f466ec80144fe](../.symplex-campaign/artifacts/cc20e5e0136c1ffb7b521b48d0b56103ff048e923bfc72942b41fc8e04f2937e.json).
- Cumulative case usage: $9.215323 / $10.00; scope change from that start through this snapshot (including later probes): $2.955248. Log symplex-data_center-v7.jsonl has 2 line(s); runner logs emit starting/completion, not each internal action.
- Milestones recorded since that attempt began:
  - problem_dna: none recorded.
  - complex_system: none recorded.
  - experiment_protocol: none recorded.
  - compute_package: none recorded.
  - execution_assessment: none recorded.
  - numerical_verification: none recorded.
  - experiment_comparison: none recorded.
  - model_critique: none recorded.
  - decision_brief: none recorded.
- Latest reusable records across all attempts: complex_system: [complex_system_2674d583b1cc4bea](../.symplex-campaign/artifacts/514e5bbe8c8692320ab56d0903b8fd558c0129285e3cd87e09821d25b10fef48.json); experiment_protocol: none recorded; compute_package: none recorded; decision_brief: none recorded.
- Latest saved delivery: [deliverable_c618b84400094920](../.symplex-campaign/artifacts/db76c462ed1831eb72e9e97c457bd96c1c755d4def188b28fad1145f20238394.json); status=dependency_gap; linked compute packages=0.
- Latest requested actions: plan_experiment ([agent_action_194de8eeeb944c18](../.symplex-campaign/artifacts/fc6d08d6cfa7b3ee3bfca61d45e399d44663b1a2d6468ce6ed7e230baea184aa.json)), plan_experiment ([agent_action_9478cceb5e534aa9](../.symplex-campaign/artifacts/e873a004626f7c6c3b74ff3780bdb8f3d81074137fdc876044adabb15dc38b7e.json)), deliver ([agent_action_84412b5b2c4c4dd6](../.symplex-campaign/artifacts/089a5bb9f2e59e66559227d02c5a3fd9d7a97f7247d7938810f7a07329fa0e1c.json)).
- Source use: 0 new research note(s), 1 lifetime note(s), 2 distinct cited URLs in retained notes. New notes: none recorded. Raw sources: none recorded.
- SDK role execution across this case: 0 native responses returned through Agent/Runner; 1 failed or were denied. A returned proposal still requires host contract acceptance.
  - [sdk_agent_run_8a6ecf78494e45bc](../.symplex-campaign/artifacts/b07d85232ec450741c2937883e0d9b0dc5b4beb03a8bed275bc01f601c1956e6.json): ExperimentProtocol, status=failed, tool invocations=0, reported usage=False; host rejection=none recorded.
- URLs actually cited in retained research notes (not a raw-file acquisition claim): https://eta-publications.lbl.gov/sites/default/files/2024-12/lbnl-2024-united-states-data-center-energy-usage-report.pdf, https://www.azwater.gov/phoenix-ama-groundwater-supply-updates.
- Steering: 2 retained directions; 0 added since that attempt began. New directions: none recorded. Latest planner/steering acknowledgement: [planner_input_ack_f941a1723a8c4ead](../.symplex-campaign/artifacts/9aa8d66ccc43dc5207ea049712073e9ac0bbcf20c096e54ee9e408d6728a90e5.json). New directions not yet in a recorded planning acknowledgement: none recorded.
- Explicit evolution: 0 evolve_model action(s); 0 candidate designs; 0 method candidates. Search-archive artifacts alone are not an executed evolutionary loop.
- Failures/rejections since that attempt began: 6 recorded.
  - [model_failure_31bdee7c0ed34ac4](../.symplex-campaign/artifacts/bcff140b3c344829f17054db94b7ad54c9ab956530638f5f32f2f2630fba355e.json): Model response did not complete: incomplete; reason=max_output_tokens; requested output token limit=3500.
  - [agent_result_aaf9a315f0324744](../.symplex-campaign/artifacts/c08d66e719fc719a6a83ba88d2505726e3fd7b3a7d593b7cf9bca4bcef020134.json): Model call failed: Invalid: Model response did not complete: incomplete; reason=max_output_tokens; requested output token limit=3500; failure artifact model_failure_31bdee7c0ed34ac4.
  - [model_failure_f984263ca6cb4c94](../.symplex-campaign/artifacts/12159337a6dba7fd154b82ef1312870e8888dada7196497deab3b07f1cf3ca0e.json): Model response did not complete: incomplete; reason=max_output_tokens; requested output token limit=3500.
  - [agent_result_7798577c6c5345e6](../.symplex-campaign/artifacts/00f471faf62f07b75afec0ffbb9d27eb405376da815f89bf929bee2754a8101f.json): Model call failed: Invalid: Model response did not complete: incomplete; reason=max_output_tokens; requested output token limit=3500; failure artifact model_failure_f984263ca6cb4c94.
  - [action_rejection_fac6d9371ad141fa](../.symplex-campaign/artifacts/c22bd6413d6c66c73b57c3dc1e2804d63f7742bef4f7284893d5e9352e88f636.json): Model revision cap reached.
  - [model_failure_c04cae0236294c1b](../.symplex-campaign/artifacts/139e89a23a07745035e20e98933e78154a0a10da72c05a726ed664e227afae2d.json): APIConnectionError.
- Earlier runtime/contract failures retained (12 records):
  - [model_failure_1c39c48f7b9b4202](../.symplex-campaign/artifacts/aa820f37a481befd52a6130c73dfdec01027fe37aafb11595b12ccb96236b6b2.json): Invalid.
  - [failure_e3807f03c1824998](../.symplex-campaign/artifacts/31f305dedbdbdf812565c07b036275ee0ad50cf4be7c14b97120e5a611c6fdfa.json): Model call failed: Invalid; see persisted failure record.
  - [model_failure_f9d9340b7f0046a6](../.symplex-campaign/artifacts/421cd087a8f673f93f6df5fc6f3beb8f2f0ca2e7d65faec62375737816f84743.json): Model response did not complete: incomplete.
  - [agent_result_eefb8978a3ba407e](../.symplex-campaign/artifacts/4c407ccec08562a86e96f499f87a31f8eabc460ac074be3fdcb699262f7ff73c.json): Model call failed: Invalid; see persisted failure record.
  - [model_failure_805d2ff4c3de4590](../.symplex-campaign/artifacts/9b25c0b6ed63cf492c7ee66af42e6ddca69bcd60a14c455d64ed99e47c49872b.json): Model response did not complete: incomplete.
  - [agent_result_1b8432ca80894829](../.symplex-campaign/artifacts/dc199a24972fabf363544b420e1f3ae7cddedb01471cf39684b75dff327e6bd2.json): Model call failed: Invalid; see persisted failure record.
  - [action_rejection_0748056b01b44f24](../.symplex-campaign/artifacts/1b0dcc4de6cc865c17c53947f6640894eb36b3a2318d14dc759018dd76d77932.json): Model revision cap reached.
  - [model_failure_7654dce10fd94dc7](../.symplex-campaign/artifacts/2fcf0cc1a847be7dbeeba6ab8ba257a3f3c9f3dfff83394afe614d8f9eef5a67.json): APITimeoutError.
  - [agent_result_4d7a80dd76d844de](../.symplex-campaign/artifacts/fdb885984013416cdef16b39754a02509593f20a9bb161fdd8855bef39ae6bec.json): Model call failed: APITimeoutError; failure artifact model_failure_7654dce10fd94dc7.
  - [agent_result_d083f0024b4343e6](../.symplex-campaign/artifacts/9c40434e66ef1ae71646a504854adcd85765fb495c5e6f8becd463102aea99d9.json): Context exceeds the bounded investigation envelope.
  - [agent_result_5b734063fb9d424f](../.symplex-campaign/artifacts/d670ee7c1d3f87ce12f8361fddb00961eafcfbfbde2d892433121b40594eb778.json): Context exceeds the bounded investigation envelope.
  - [failure_55b12759e3304212](../.symplex-campaign/artifacts/64866b1c77ce6d35db6cfd46a7b61f9612835b1acaf870b372b06599f14adb7c.json): 1 validation error for ArtifactRead limit   Input should be less than or equal to 8000 [type=less_than_equal, input_value=18000, input_type=int]     For further information visit https://errors.pydantic.dev/2.13/v/less_than_equal.

## hudson_ecology

- Problem: [workspace_problem_91a2513b8d2d4f6e](../.symplex-campaign/artifacts/422307ebf83429d0d4b342f018cde65a24355d047b462472603297f7e5770a39.json); latest attempt: [campaign_attempt_a2e5645a7b654335](../.symplex-campaign/artifacts/82189460c455791b22bc777d15cead65b6779a4e7e9b7caa7da8c45942800ddc.json). Latest recorded job status since that attempt began: budget-exhausted; attempt end: [campaign_attempt_end_5387794b9bb347f0](../.symplex-campaign/artifacts/e603b59f6bf1920b64818f9797bef10481f318d334ba0fd0f21f95e8fc061ed3.json).
- Cumulative case usage: $8.973340 / $10.00; scope change from that start through this snapshot (including later probes): $0.736025. Log symplex-hudson_ecology-v7.jsonl has 2 line(s); runner logs emit starting/completion, not each internal action.
- Milestones recorded since that attempt began:
  - problem_dna: [problem_dna_9c2bb1c633724e0c](../.symplex-campaign/artifacts/847f5e3f3ecf1bd0a7aca7059041a619655e615ff3cffba22ca007b9d0da3851.json).
  - complex_system: none recorded.
  - experiment_protocol: none recorded.
  - compute_package: none recorded.
  - execution_assessment: none recorded.
  - numerical_verification: none recorded.
  - experiment_comparison: none recorded.
  - model_critique: none recorded.
  - decision_brief: none recorded.
- Latest reusable records across all attempts: complex_system: [complex_system_b1f42532575247dc](../.symplex-campaign/artifacts/d597dc53b77409347e067828de80bd66812138ceed13f66a125953060ad0ac8b.json); experiment_protocol: none recorded; compute_package: none recorded; decision_brief: none recorded.
- Latest requested actions: inspect_artifact ([agent_action_b09a1a132def4876](../.symplex-campaign/artifacts/4210c9a0f04ee3ad002c0413aa2c83ab41322d9e672a2d1f765bc58da61c3720.json)), inspect_artifact ([agent_action_3fa6f955da874a85](../.symplex-campaign/artifacts/728a1ef1ac133f81a5d0e801e336b632bc277dc4100e766726b74b46ac0e4262.json)), plan_experiment ([agent_action_8581bd9b7a3343e3](../.symplex-campaign/artifacts/cd730e4b56cef71ab6c35bbf94e8e36ebc814993f774a8d3ba4bd00924000b54.json)).
- Source use: 0 new research note(s), 1 lifetime note(s), 5 distinct cited URLs in retained notes. New notes: none recorded. Raw sources: none recorded.
- SDK role execution across this case: 7 native responses returned through Agent/Runner; 2 failed or were denied. A returned proposal still requires host contract acceptance.
  - [sdk_agent_run_9b5b1adc5d794d01](../.symplex-campaign/artifacts/4f773d17be320c9f269cd19f6fcf64cba214a8f48451fa5a760d9325dcb7a097.json): ExperimentProtocol, status=failed, tool invocations=0, reported usage=False; host rejection=none recorded.
  - [sdk_agent_run_e9f0746b27d94dbb](../.symplex-campaign/artifacts/4ddc2b56dc09c49cc92709fa054e1e889c2cd1a29f29828745ce9549b441fe33.json): ExperimentProtocol, status=returned_to_host, tool invocations=1, reported usage=True; host rejection=[proposal_rejection_1fb906974fa54cac](../.symplex-campaign/artifacts/4b33ebbd49af05c2859588d135c30be3118c26afb2442122579d753556354593.json).
  - [sdk_agent_run_e44e48821e244c98](../.symplex-campaign/artifacts/f19d28c80ddbc8ac0da0a498f7804ba91950f2b79a0b923c5363115af3dcd7ae.json): ExperimentProtocol, status=returned_to_host, tool invocations=1, reported usage=True; host rejection=[proposal_rejection_7254ffc04f2e4254](../.symplex-campaign/artifacts/5692732233dc05477cc1d9f951e115c05d268e59c60eedd1fbe8a3d2ffb87eaa.json).
  - [sdk_agent_run_9a68c7c4b844498c](../.symplex-campaign/artifacts/306a3df9b6bf2d7e371865947533c990a3dc393c78957c54874e03dd58b84f07.json): ProblemDNA, status=returned_to_host, tool invocations=1, reported usage=True; host rejection=none recorded.
  - [sdk_agent_run_f149242da2a443ae](../.symplex-campaign/artifacts/b85645b0022547a88400c44491c42209b5bd1d1729fb0f11a909c84fd61b67d5.json): NextStep, status=returned_to_host, tool invocations=1, reported usage=True; host rejection=none recorded.
  - [sdk_agent_run_8cc67a7e2e164896](../.symplex-campaign/artifacts/8a3613dd20e94f7e9fac855d0fb9834b88fb4d86dd31d21e48176f126ea27072.json): NextStep, status=returned_to_host, tool invocations=1, reported usage=True; host rejection=none recorded.
  - [sdk_agent_run_4bcae17c08734abb](../.symplex-campaign/artifacts/eb01522effb036dfe6ec7c79baf7ce3e7e662e2b6db12dea3b4e83c548010599.json): NextStep, status=returned_to_host, tool invocations=1, reported usage=True; host rejection=none recorded.
  - [sdk_agent_run_854550d4f5de4e3e](../.symplex-campaign/artifacts/40e0c93bdebcf8562ffb3638ee91f0fc19187f4346826349cf92c06625fb8698.json): NextStep, status=returned_to_host, tool invocations=1, reported usage=True; host rejection=none recorded.
  - [sdk_agent_run_1f461c7b67214915](../.symplex-campaign/artifacts/2c0894cf5c6248694d3768857d447a32ebd2a4e90064e549e6d958ea09348bcc.json): ExperimentProtocol, status=failed, tool invocations=0, reported usage=False; host rejection=none recorded.
- URLs actually cited in retained research notes (not a raw-file acquisition claim): https://dec.ny.gov/environmental-protection/water/water-quality/dmr-manual-for-submitting-completing-discharge-monitoring-report?utm_source=openai, https://dec.ny.gov/environmental-protection/water/water-quality/sewage-pollution-right-to-know?utm_source=openai, https://hrecos.org/request-data/, https://waterdata.usgs.gov/nwis/uv/?PARAmeter_cd=00400&site_no=01359165&utm_source=openai, https://www.albanypoolcso.org/water-quality/.
- Steering: 1 retained directions; 0 added since that attempt began. New directions: none recorded. Latest planner/steering acknowledgement: [planner_input_ack_0de53094378043b4](../.symplex-campaign/artifacts/f7cb9635a3ad5c2322fe9121a052de85bb07d4d402c1c4564601a7ba7eb7a0de.json). New directions not yet in a recorded planning acknowledgement: none recorded.
- Explicit evolution: 0 evolve_model action(s); 0 candidate designs; 0 method candidates. Search-archive artifacts alone are not an executed evolutionary loop.
- Failures/rejections since that attempt began: 1 recorded.
  - [failure_9cf15353e5074547](../.symplex-campaign/artifacts/b3fef486be8e4df00dc44355f621db11945fb39ffb82e751fa50307621a2df07.json): Scope allocation exhausted: usd.
- Earlier runtime/contract failures retained (17 records):
  - [model_failure_1309b246545344a0](../.symplex-campaign/artifacts/4f570054aff1243572a5cecea18012108057e93789eb4b482fc57a12e3ab46e1.json): Invalid.
  - [failure_e90553f7dd3e4e6f](../.symplex-campaign/artifacts/31f305dedbdbdf812565c07b036275ee0ad50cf4be7c14b97120e5a611c6fdfa.json): Model call failed: Invalid; see persisted failure record.
  - [model_failure_9afb4234d4f24421](../.symplex-campaign/artifacts/9c2d490184f07801d1fe8442ba3ebf64c3d9a583a6bfdc1bf812df0ff5b914df.json): APITimeoutError.
  - [agent_result_3982e750074b45b8](../.symplex-campaign/artifacts/c2eaf416261593c44b1d86061d6777d4c58876e4f75baeb1c1f29be3a8fea8b5.json): Model call failed: APITimeoutError; failure artifact model_failure_9afb4234d4f24421.
  - [proposal_rejection_ae6cca0c562b4a81](../.symplex-campaign/artifacts/d82199616ec0fc312590bc8b1a66258706197a414781da30d0a4d099d3d7d179.json): 1 validation error for ComplexSystemSpec   Value error, Unknown/event timing requires an explicit event_mapping alignment [type=value_error, input_value={'title': 'Albany–Castl...sure is the endpoint.']}, input_type=dict]     For further information visit https://errors.pydantic.dev/2.13/v/value_error.
  - [agent_result_fc8ade1f998743a8](../.symplex-campaign/artifacts/ceafb05cc8458c05d21c11f298a1202256400929cc3e6ccfc340aee8830f40dc.json): Context exceeds the bounded investigation envelope.
  - [agent_result_6c5d8b7f25dd493b](../.symplex-campaign/artifacts/7d88eb58500d6b6ecdf475e3ed20cd7101ee1e105c8557b8e6054dfeb67ef329.json): Context exceeds the bounded investigation envelope.
  - [agent_result_29ec2a6b3fc648d8](../.symplex-campaign/artifacts/10712865e82ccbd15ec4459c1571efebf8d01c202c29b7362b0d3ed802e7cba2.json): Context exceeds the bounded investigation envelope.
  - [failure_31afb14665ca4890](../.symplex-campaign/artifacts/858e9559c66a536afee73d1592af7a4c42f358be86cc48b56eade252426e1791.json): 1 validation error for ArtifactRead limit   Input should be less than or equal to 8000 [type=less_than_equal, input_value=24000, input_type=int]     For further information visit https://errors.pydantic.dev/2.13/v/less_than_equal.
  - [model_failure_aea20d16b8c54f74](../.symplex-campaign/artifacts/0379c52d5d4c66bb500901e6801984a182b36b1f7c4d2f4d0ad684003cd5db75.json): Model response did not complete: incomplete; reason=max_output_tokens; requested output token limit=3500.
  - [agent_result_7f0d8a533bd5419c](../.symplex-campaign/artifacts/a3e54217a101fc864f12ed81f4d5efb84dd2d7e85af4436404ab8075d0fd6965.json): Model call failed: Invalid: Model response did not complete: incomplete; reason=max_output_tokens; requested output token limit=3500; failure artifact model_failure_aea20d16b8c54f74.
  - [model_failure_7ef55f3a920f499b](../.symplex-campaign/artifacts/e79ef0947577792eb2fbd17025e639bdb871f769a5d80087065e6eb6fbe16129.json): Model response did not complete: incomplete; reason=max_output_tokens; requested output token limit=3500.
  - [agent_result_15388e9825f14bf2](../.symplex-campaign/artifacts/50cde0c8a945ac7cd9bc125ecc6c726035d73769522111ab9ac2fb8ce271ac95.json): Model call failed: Invalid: Model response did not complete: incomplete; reason=max_output_tokens; requested output token limit=3500; failure artifact model_failure_7ef55f3a920f499b.
  - [action_rejection_35a8e97220714ab9](../.symplex-campaign/artifacts/6feca73e022749b57cbaa34e964afe603d0dc4550035b62c81b438c0beafcae0.json): Model revision cap reached.
  - [model_failure_ae759fa4ade145a6](../.symplex-campaign/artifacts/4b5e29b3fe1f71e97673cd8e7201d9b98a7e4d09412cd8e5b5e3fdec4540473a.json): APIConnectionError.
  - [proposal_rejection_1fb906974fa54cac](../.symplex-campaign/artifacts/4b33ebbd49af05c2859588d135c30be3118c26afb2442122579d753556354593.json): 1 validation error for ExperimentProtocol numeric_checks.0   Value error, Time must be distinct from value and group columns [type=value_error, input_value={'id': 'finite', 'operati...xtrema must be finite.'}, input_type=dict]     For further information visit https://errors.pydantic.dev/2.13/v/value_error.
  - [proposal_rejection_7254ffc04f2e4254](../.symplex-campaign/artifacts/5692732233dc05477cc1d9f951e115c05d268e59c60eedd1fbe8a3d2ffb87eaa.json): 3 validation errors for ExperimentProtocol comparison_controls   String should have at least 1 character [type=string_too_short, input_value='', input_type=str]     For further information visit https://errors.pydantic.dev/2.13/v/string_too_short uncertainty_plan   String should have at least 1 character [type=string_too_short, input_value='', input_type=str]     For further information visit https://errors.pydantic.dev/2.13/v/string_too_short stopping_rule   String should have at least 1 charac.

## midterms

- Problem: [workspace_problem_d8962462d8fa43db](../.symplex-campaign/artifacts/f14c1d26f32219c7cc4edb63808189dd317f67dbb1ea45db8a12198059fe391c.json); latest attempt: [campaign_attempt_08d48d87ef82467c](../.symplex-campaign/artifacts/4bda5906a3592a2cbf50db34e63f47767f2cbbdfed2f6646a6fa71957407a0dc.json). Latest recorded job status since that attempt began: succeeded; attempt end: [campaign_attempt_end_76b69f2ba7824d5f](../.symplex-campaign/artifacts/05ca9cc495de41522060da5c197c4eeb008834114ffd2d09cc324f6aa0f7f81f.json).
- Cumulative case usage: $4.944440 / $10.00; scope change from that start through this snapshot (including later probes): $3.459815. Log symplex-midterms-v5.jsonl has 2 line(s); runner logs emit starting/completion, not each internal action.
- Milestones recorded since that attempt began:
  - problem_dna: [problem_dna_6782c41182224050](../.symplex-campaign/artifacts/08b95d6649998100a0b2f9e48760edcc30df74688c9f547a0305c742a2aed38b.json).
  - complex_system: none recorded.
  - experiment_protocol: none recorded.
  - compute_package: none recorded.
  - execution_assessment: none recorded.
  - numerical_verification: none recorded.
  - experiment_comparison: none recorded.
  - model_critique: none recorded.
  - decision_brief: none recorded.
- Latest reusable records across all attempts: complex_system: none recorded; experiment_protocol: none recorded; compute_package: none recorded; decision_brief: none recorded.
- Latest saved delivery: [deliverable_3e6ae06afecc47be](../.symplex-campaign/artifacts/faa7c840ed876aaf1dc7f3d4d8466230c0e14df73ae77c15390b8f467a18ef86.json); status=dependency_gap; linked compute packages=0.
- Latest requested actions: represent_system ([agent_action_773ea9da049a4a33](../.symplex-campaign/artifacts/3d4a7680445a0e1ab66df3f0910dba6c239d9f29090d91b58c5442195111668d.json)), design_solution ([agent_action_cb390e74ea9d46ee](../.symplex-campaign/artifacts/7b734efa6b3d548eeec717212d74d0626331a24b7feac01fe78f2f610e0e55e6.json)), deliver ([agent_action_f16bd02807144744](../.symplex-campaign/artifacts/de69b8e55ed898dde4cdd419ad2b52f683cf0521c4c8b69a714634046a454e5a.json)).
- Source use: 0 new research note(s), 1 lifetime note(s), 5 distinct cited URLs in retained notes. New notes: none recorded. Raw sources: none recorded.
- URLs actually cited in retained research notes (not a raw-file acquisition claim): https://uscode.house.gov/view.xhtml?edition=prelim&num=0&req=granuleid%3AUSC-prelim-title2-section7&utm_source=openai, https://www.fec.gov/introduction-campaign-finance/election-results-and-voting-information/?utm_source=openai, https://www.fec.gov/introduction-campaign-finance/election-results-and-voting-information/federal-elections-2020/, https://www.fec.gov/introduction-campaign-finance/election-results-and-voting-information/federal-elections-2022/, https://www.house.gov/representatives.
- Steering: 1 retained directions; 0 added since that attempt began. New directions: none recorded. Latest planner/steering acknowledgement: [planner_input_ack_ea984b113f5741a4](../.symplex-campaign/artifacts/843333fdd179f5fc6cd0242d36061e5d0dd61c2564e483eba2f6918ac0e0f3ab.json). New directions not yet in a recorded planning acknowledgement: none recorded.
- Explicit evolution: 0 evolve_model action(s); 0 candidate designs; 0 method candidates. Search-archive artifacts alone are not an executed evolutionary loop.
- Failures/rejections since that attempt began: 5 recorded.
  - [model_failure_e4bb8693a5b94eee](../.symplex-campaign/artifacts/9e3adbd64e878b386543df377eb8b1d70d845dd85efb4251205e6ccdae067b00.json): APITimeoutError.
  - [agent_result_673f5c59e9634aa6](../.symplex-campaign/artifacts/a6f4c4f360d5d5c94512cd0774a1e20505c0d3464c3852745af0f40cc5a0a0ab.json): Model call failed: APITimeoutError; failure artifact model_failure_e4bb8693a5b94eee.
  - [model_failure_8bf7b0d8ede942f3](../.symplex-campaign/artifacts/ebf66b380a8a1199df4a79cd95b386c03e624aa2378a339dd1b30cc9ecbc5932.json): APITimeoutError.
  - [agent_result_396ec2b3b23243ec](../.symplex-campaign/artifacts/029e0ac5b05d68abbe01841a5c197b87167311beb2ea07d00274c764165716e4.json): Model call failed: APITimeoutError; failure artifact model_failure_8bf7b0d8ede942f3.
  - [action_rejection_c4d9b942cfef4edb](../.symplex-campaign/artifacts/3e2eda5b401c9185b2303d25313531e61d1936e79fe33038fd98b01ce4f885f5.json): Model revision cap reached.
- Earlier runtime/contract failures retained (2 records):
  - [model_failure_fc9f8ff7f74d41be](../.symplex-campaign/artifacts/340703cbbcf9b1b8e8274066f7fb68a385a3a2376dae8b6217565f53ab231273.json): InternalServerError.
  - [failure_1064c0c578614bff](../.symplex-campaign/artifacts/828c118c725138590269d6c545d150bf9dc96d4caaec75310af0bccc3d7e3d31.json): Model call failed: InternalServerError; failure artifact model_failure_fc9f8ff7f74d41be.

## protein

- Problem: [workspace_problem_fabb1c631ab14f47](../.symplex-campaign/artifacts/b646ead4cf2ac90ee9fe20df3ca28150cc8197b5e540c2d8905c9b084a51e6a4.json); latest attempt: [campaign_attempt_2224443ef0a447bb](../.symplex-campaign/artifacts/d4697ccb09ff53ece1200119efe729c3a895b0ab64b371a6f279075f0df967d7.json). Latest recorded job status since that attempt began: succeeded; attempt end: [campaign_attempt_end_39f8b739e58d478e](../.symplex-campaign/artifacts/9e0b2f1fbd734c5aa8037f4023666fa498a3824133c79793afbd224ab0d8704f.json).
- Cumulative case usage: $8.477278 / $10.00; scope change from that start through this snapshot (including later probes): $2.770085. Log symplex-protein-v7.jsonl has 2 line(s); runner logs emit starting/completion, not each internal action.
- Milestones recorded since that attempt began:
  - problem_dna: none recorded.
  - complex_system: none recorded.
  - experiment_protocol: [experiment_protocol_fd4ff883830e4411](../.symplex-campaign/artifacts/632ad07feff9a62288c71e4d55a626848b5e7dd4f58cbd9e19f603a20cf2c0f8.json).
  - compute_package: none recorded.
  - execution_assessment: none recorded.
  - numerical_verification: none recorded.
  - experiment_comparison: none recorded.
  - model_critique: none recorded.
  - decision_brief: [decision_brief_bbd35b965126403c](../.symplex-campaign/artifacts/010f4c4a1747610fea9e0f55f8c396e583c58dd2e24d3c644e0c1459c3e3ce83.json).
- Latest reusable records across all attempts: complex_system: [complex_system_4729bd5558f74aee](../.symplex-campaign/artifacts/c573fa58a2a63f0c1b26126854546b01a1e0f235c44d95c8ca1eab93530d2b0c.json); experiment_protocol: [experiment_protocol_fd4ff883830e4411](../.symplex-campaign/artifacts/632ad07feff9a62288c71e4d55a626848b5e7dd4f58cbd9e19f603a20cf2c0f8.json); compute_package: none recorded; decision_brief: [decision_brief_bbd35b965126403c](../.symplex-campaign/artifacts/010f4c4a1747610fea9e0f55f8c396e583c58dd2e24d3c644e0c1459c3e3ce83.json).
- Latest saved delivery: [deliverable_2e3e79891f904dbc](../.symplex-campaign/artifacts/67d4c37ad8d29ca4a498feede0705c83b4ac91382beb91abbabe67997f0d656c.json); status=dependency_gap; linked compute packages=0.
- Latest requested actions: synthesize_evidence ([agent_action_fa4a0c621fb246a8](../.symplex-campaign/artifacts/83280056a4b3a846e35dc4cae21de438bb97fb07058bd05718a745fbdcfaeabd.json)), build_outcome ([agent_action_8396810b8e9d4730](../.symplex-campaign/artifacts/2092ef6b49bb55cb4511fde69eade2716d9e1c8aa94e37db09c9e3e98aeeda9f.json)), deliver ([agent_action_f215649f3a8540ad](../.symplex-campaign/artifacts/1152ad1e23e4eb80c82b520bd1f4b6497d2bc75ce55c8e109db243187deef586.json)).
- Source use: 1 new research note(s), 2 lifetime note(s), 2 distinct cited URLs in retained notes. New notes: [evidence_note_fc8697786124451e](../.symplex-campaign/artifacts/2a5b75df1feafa24b9f27723a00eab4848e8c7657a27f1d3d1453277e1743626.json). Raw sources: none recorded.
  - [evidence_note_fc8697786124451e](../.symplex-campaign/artifacts/2a5b75df1feafa24b9f27723a00eab4848e8c7657a27f1d3d1453277e1743626.json): provider completeness=complete; narrative mentions partial/gap=False. The completeness flag is not evidence coverage or claim validation.
- URLs actually cited in retained research notes (not a raw-file acquisition claim): https://journals.plos.org/plosone/article?id=10.1371%2Fjournal.pone.0141169, https://pubmed.ncbi.nlm.nih.gov/30886367/?utm_source=openai.
- Steering: 1 retained directions; 0 added since that attempt began. New directions: none recorded. Latest planner/steering acknowledgement: [planner_input_ack_2f1ef9aaf5ac41c1](../.symplex-campaign/artifacts/e246eada142a59cb566f6808a5d2473867b50e1589d4ae637c218240e2642f11.json). New directions not yet in a recorded planning acknowledgement: none recorded.
- Explicit evolution: 0 evolve_model action(s); 0 candidate designs; 0 method candidates. Search-archive artifacts alone are not an executed evolutionary loop.
- Failures/rejections since that attempt began: 2 recorded.
  - [proposal_rejection_fca3db1b946e4a3e](../.symplex-campaign/artifacts/c60d929d33997ff83973d41ffbcc524f2136eba7d085c7b9ea454da90325da82.json): 1 validation error for ExperimentProtocol numeric_checks.0   Value error, Time must be distinct from value and group columns [type=value_error, input_value={'id': 'trajectory_finite...oordinates and states.'}, input_type=dict]     For further information visit https://errors.pydantic.dev/2.13/v/value_error.
  - [agent_result_c14b4218fac448b9](../.symplex-campaign/artifacts/0c46c0320fff3298fff6f524f6d703be93868cb4ae77919e6fb54917e1d643cb.json): Context exceeds the bounded investigation envelope.
- Earlier runtime/contract failures retained (11 records):
  - [model_failure_b25dcde6ee8043e8](../.symplex-campaign/artifacts/0f96daae2df7ab527c8bd844e92e6954d6bddbebc9f2b4b27f049a95914f1586.json): Invalid.
  - [failure_c0df6e4488ed46db](../.symplex-campaign/artifacts/31f305dedbdbdf812565c07b036275ee0ad50cf4be7c14b97120e5a611c6fdfa.json): Model call failed: Invalid; see persisted failure record.
  - [model_failure_edb0cd4391694bd8](../.symplex-campaign/artifacts/ffda35832816646c433b1d4395bfd554ed3b8f9404f737a5d8fe7634601d2097.json): APITimeoutError.
  - [agent_result_8be110858bf0488f](../.symplex-campaign/artifacts/e6b84dc3d8dc00ddabd62a399fb2007df154056714fcf0a04e55da7c35e97cf0.json): Model call failed: APITimeoutError; failure artifact model_failure_edb0cd4391694bd8.
  - [proposal_rejection_de02d5e79dac414a](../.symplex-campaign/artifacts/2c4549172fe5a0f9fc859f4cd402779a4f09812bf6f8534f032810d2d59e38ae.json): 1 validation error for ComplexSystemSpec   Value error, Port rec_e must use its state's declared time scale [type=value_error, input_value={'title': 'Trypsin assay-... mechanism selection.']}, input_type=dict]     For further information visit https://errors.pydantic.dev/2.13/v/value_error.
  - [agent_result_32ec7891a4464ab1](../.symplex-campaign/artifacts/44216e786163f94c8dddab420ac57190680e776e683ae0b8a8b016eb1c40f50b.json): Context exceeds the bounded investigation envelope.
  - [action_rejection_b67ca119d8cd47b2](../.symplex-campaign/artifacts/fc747a1c2c11bcd04d54cbfc798fa9de91cb18464394b8f973efb79d1d2dafa8.json): Model revision cap reached.
  - [proposal_rejection_a145c2e7a6904eec](../.symplex-campaign/artifacts/574d4be52acd4616420a56680d9b6011dd5d16d7db9c620e426cfecedc877b88.json): 1 validation error for ComplexSystemSpec   Value error, Unknown/event timing requires an explicit event_mapping alignment [type=value_error, input_value={'title': 'Trypsin activi...r this research lead?']}, input_type=dict]     For further information visit https://errors.pydantic.dev/2.13/v/value_error.
  - [agent_result_dd0cf8fa92b04d8c](../.symplex-campaign/artifacts/76888d53988aafa5596c582176a20454957ca9348cb5c06a13034b84bc286835.json): Context exceeds the bounded investigation envelope.
  - [agent_result_8982090351154915](../.symplex-campaign/artifacts/7bbac34f46963727fa22f4da96fd88ed70f67757ca0f01d9f7fde0d27f515a89.json): Context exceeds the bounded investigation envelope.
  - [failure_5d0de8fa399e4892](../.symplex-campaign/artifacts/858e9559c66a536afee73d1592af7a4c42f358be86cc48b56eade252426e1791.json): 1 validation error for ArtifactRead limit   Input should be less than or equal to 8000 [type=less_than_equal, input_value=24000, input_type=int]     For further information visit https://errors.pydantic.dev/2.13/v/less_than_equal.

## i_model_routing

- Problem: [workspace_problem_452d2788d1e6408e](../.symplex-campaign/artifacts/7ca84c26c9ecdccd27484cd470907c8c1b69179a313c94c897bb67eae12c99eb.json); latest attempt: [campaign_attempt_5459e6c406454417](../.symplex-campaign/artifacts/b37bfe52bfd0d4643dcb2a7fa0e01c9f5691ef995bf96bde61a739e6d475588d.json). Latest recorded job status since that attempt began: succeeded; attempt end: [campaign_attempt_end_d7036a1b54074ce8](../.symplex-campaign/artifacts/6c9a238168218aeff0c16ce5a7c672de2d16276fa9192e8721df5f68a7d5d6c4.json).
- Cumulative case usage: $4.044952 / $5.00; scope change from that start through this snapshot (including later probes): $2.747315. Log symplex-i_model_routing-v2.jsonl has 2 line(s); runner logs emit starting/completion, not each internal action.
- Milestones recorded since that attempt began:
  - problem_dna: none recorded.
  - complex_system: [complex_system_f1be530f7ac942f3](../.symplex-campaign/artifacts/80d70f212f52cc7a4b70236c8ea177d1e44b3f2bddd55932ee4e77233cea385a.json).
  - experiment_protocol: none recorded.
  - compute_package: none recorded.
  - execution_assessment: none recorded.
  - numerical_verification: none recorded.
  - experiment_comparison: none recorded.
  - model_critique: none recorded.
  - decision_brief: [decision_brief_84a88b30979849d7](../.symplex-campaign/artifacts/a8187e5a8dbe458a83d2703785838fccde47ba46cc9ffd735acb2b811dda2902.json).
- Latest reusable records across all attempts: complex_system: [complex_system_f1be530f7ac942f3](../.symplex-campaign/artifacts/80d70f212f52cc7a4b70236c8ea177d1e44b3f2bddd55932ee4e77233cea385a.json); experiment_protocol: none recorded; compute_package: none recorded; decision_brief: [decision_brief_84a88b30979849d7](../.symplex-campaign/artifacts/a8187e5a8dbe458a83d2703785838fccde47ba46cc9ffd735acb2b811dda2902.json).
- Latest saved delivery: [deliverable_4caf04f0c4c84613](../.symplex-campaign/artifacts/52d9609a408175308bf3cf08feed6dcc50ffd32adf96f97b2a3e11bd2c4e80c3.json); status=dependency_gap; linked compute packages=0.
- Latest requested actions: plan_experiment ([agent_action_13aadb8f944a408f](../.symplex-campaign/artifacts/c83aac36e2fcec98c411f7b1ebadac58e132cbd0f0fefed69dee495366c83610.json)), build_outcome ([agent_action_5133641f36454bf8](../.symplex-campaign/artifacts/7a5109ecefcebd548c6ca8d3b635022c445a60d3848438fa787e9557a63d8ce4.json)), deliver ([agent_action_20288999b3e14f2a](../.symplex-campaign/artifacts/db98a0522bb322f4a3c8942bd184d153de8f9192e20000818c966b1fc2abf484.json)).
- Source use: 0 new research note(s), 2 lifetime note(s), 6 distinct cited URLs in retained notes. New notes: none recorded. Raw sources: none recorded.
- URLs actually cited in retained research notes (not a raw-file acquisition claim): https://arxiv.org/abs/2406.18665?utm_source=openai, https://github.com/lm-sys/RouteLLM/blob/main/routellm/evals/benchmarks.py, https://github.com/lm-sys/RouteLLM/tree/main/routellm/evals, https://github.com/lm-sys/routellm?utm_source=openai, https://github.com/withmartian/routerbench, https://huggingface.co/datasets/withmartian/routerbench/tree/main.
- Steering: 0 retained directions; 0 added since that attempt began. New directions: none recorded. Latest planner/steering acknowledgement: [planner_input_ack_4ecd66bc4484438e](../.symplex-campaign/artifacts/c85387f0c55ce93f3fcf2c9c2357582e19ffd51303cae054f19e1b6c46b080e2.json). New directions not yet in a recorded planning acknowledgement: none recorded.
- Explicit evolution: 0 evolve_model action(s); 0 candidate designs; 0 method candidates. Search-archive artifacts alone are not an executed evolutionary loop.
- Failures/rejections since that attempt began: 3 recorded.
  - [proposal_rejection_c42dd2aa6e744146](../.symplex-campaign/artifacts/c2ea2b2b10ed60cdf0807c73fdf773f2ad2d69f42a277dba7a86f0751415f089.json): 1 validation error for ComplexSystemSpec   Value error, Unknown/event timing requires an explicit event_mapping alignment [type=value_error, input_value={'title': 'Synthetic two-... untested extensions.']}, input_type=dict]     For further information visit https://errors.pydantic.dev/2.13/v/value_error.
  - [agent_result_399a9f49f5b14fcc](../.symplex-campaign/artifacts/ce390bb78b68e1f18c11b0641881bbc88e4bb2fc99aaf62d7ef7a99e7696202c.json): Complete primary contract and output schema exceed the request envelope; select a smaller scoped task rather than truncate the contract.
  - [agent_result_84bd1cfb34b24241](../.symplex-campaign/artifacts/47fafef9c0ccac0792feded94905b8c4380f4fa737e7d31323e7b66e8416ae0e.json): Complete primary contract and output schema exceed the request envelope; select a smaller scoped task rather than truncate the contract.
- Earlier runtime/contract failures retained (1 records):
  - [failure_6871548218db45a8](../.symplex-campaign/artifacts/a926295402da31880c5927b6db97e894033989983e402737aeed5dcecd602489.json): 1 validation error for ArtifactRead limit   Input should be less than or equal to 8000 [type=less_than_equal, input_value=12000, input_type=int]     For further information visit https://errors.pydantic.dev/2.13/v/less_than_equal.

## Interpretation limits

No retained artifact in this audit is treated as a domain answer authored by the monitor. Scenario claims are agent outputs; numerical checks, where present, test their declared fixtures and tolerances. Source bytes require provenance and structural checks; scientific promotion additionally requires suitable independent measurements and evaluation. Any unavailable, omitted or rejected input remains a gap.

A running attempt is provisional. This file should be refreshed after new source, compute, verification, comparison, critique, brief, steering acknowledgement or failure artifacts appear, and after each attempt terminates.
