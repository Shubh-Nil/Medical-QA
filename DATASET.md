# =============================================================================
# PubMedQA Dataset  —  https://huggingface.co/datasets/qiaojin/PubMedQA
# =============================================================================
#
# Subsets (all have only a `train` split):
#     "pqa_labeled"      1,000 examples      {Question - Answer - Yes/ No/ Maybe}
#     "pqa_artificial"   211,000 examples    {Question - Answer - Yes/ No/ Maybe}
#     "pqa_unlabeled"    61,200 examples     {Question - Answer} 
#
# Structure:
#     pubid                            int         Unique PubMed article ID (used for citation / traceability)
#     question                         str         The medical question (query)

#     context                          dict
#        ├─ contexts                   list[str]   PubMed abstract passages (these get embedded)
#        ├─ labels                     list[str]   Section label per passage (e.g. BACKGROUND, RESULTS)
#        ├─ meshes                     list[str]   MeSH medical subject headings
#        ├─ reasoning_required_pred    str         (for "pqa_labeled" only)
#        └─ reasoning_free_pred        str         (for "pqa_labeled" only)

#     long_answer                      str         Ground-truth answer (used for generation)
#     final_decision                   str         {Yes/ No/ Maybe}  (for "pqa_labeled", it is annotated by a human)
#                                                                    (for "pqa_artificial", it is annotated by rules)
# =============================================================================