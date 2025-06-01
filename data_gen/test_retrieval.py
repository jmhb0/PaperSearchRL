"""
python -m ipdb data_gen/test_retrieval.py
Test Retrieval Script using existing E5 Encoder

This script leverages the existing Encoder class from search_r1/search/retrieval_server.py
to load the intfloat/e5-base-v2 embedding model and compute similarity scores.

Usage:
    python data_gen/test_retrieval.py
"""

import sys
import os
import numpy as np
import torch
from typing import List, Tuple
import ipdb

# Add the search_r1 directory to the path so we can import from it
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'search_r1'))

from search.retrieval_server import Encoder


class RetrievalTester:
    """
    Simple wrapper around the existing Encoder class for testing retrieval similarities.
    """

    def __init__(self,
                 model_path: str = "intfloat/e5-base-v2",
                 use_fp16: bool = False,
                 max_length: int = 256):
        """
        Initialize the retrieval tester using the existing Encoder class.
        
        Args:
            model_path: Path or name of the model (default: intfloat/e5-base-v2)
            use_fp16: Whether to use half precision for faster inference
            max_length: Maximum token length for text encoding
        """
        print(f"🔍 Initializing Retrieval Tester with model: {model_path}")

        # Initialize the encoder using the existing class
        self.encoder = Encoder(
            model_name="e5",  # This triggers the E5-specific preprocessing
            model_path=model_path,
            pooling_method="mean",
            max_length=max_length,
            use_fp16=use_fp16)

        print(f"✅ Retrieval Tester initialized successfully!")

    def encode_queries(self, queries: List[str]) -> np.ndarray:
        """
        Encode queries using the existing encoder with is_query=True.
        
        Args:
            queries: List of query strings
            
        Returns:
            Normalized embeddings as numpy array
        """
        if isinstance(queries, str):
            queries = [queries]
        return self.encoder.encode(queries, is_query=True)

    def encode_documents(self, documents: List[str]) -> np.ndarray:
        """
        Encode documents using the existing encoder with is_query=False.
        
        Args:
            documents: List of document strings
            
        Returns:
            Normalized embeddings as numpy array
        """
        if isinstance(documents, str):
            documents = [documents]
        return self.encoder.encode(documents, is_query=False)

    def compute_similarities(self, query: str,
                             documents: List[str]) -> np.ndarray:
        """
        Compute cosine similarities between a query and multiple documents.
        
        Args:
            query: Single query string
            documents: List of document strings
            
        Returns:
            Array of cosine similarity scores (higher = more similar)
        """
        query_emb = self.encode_queries([query])  # Shape: (1, embedding_dim)
        doc_embs = self.encode_documents(
            documents)  # Shape: (n_docs, embedding_dim)

        # Compute cosine similarities (embeddings are already normalized)
        similarities = np.dot(query_emb, doc_embs.T)[0]  # Shape: (n_docs,)

        return similarities

    def rank_documents(self, query: str,
                       documents: List[str]) -> List[Tuple[int, str, float]]:
        """
        Rank documents by similarity to query.
        
        Args:
            query: Single query string
            documents: List of document strings
            
        Returns:
            List of tuples (original_index, document, similarity_score) sorted by similarity (descending)
        """
        similarities = self.compute_similarities(query, documents)

        # Create list of (index, document, score) and sort by score descending
        ranked = [(i, doc, score)
                  for i, (doc,
                          score) in enumerate(zip(documents, similarities))]
        ranked.sort(key=lambda x: x[2], reverse=True)

        return ranked


def test_retrieval_example():
    """
    Example function demonstrating how to use the existing Encoder
    for computing retrieval similarities with positive and negative documents.
    """
    print("🔍 Testing E5 Retrieval Similarities")
    print("=" * 50)

    # Initialize the retrieval tester (this uses the existing Encoder class)
    tester = RetrievalTester()

    def num_tokens(text: str) -> int:
        return len(tester.encoder.tokenizer.encode(text))

    # Example query

    # Positive documents (relevant to the query)
    if 0:
        query = "What are the symptoms of diabetes?"

        positive_documents = [
            "Diabetes symptoms include increased thirst, frequent urination, extreme fatigue, and blurred vision.",
            "Common signs of diabetes are excessive hunger, unexplained weight loss, and slow-healing sores.",
            "Type 2 diabetes often develops gradually with symptoms like increased thirst and frequent urination."
        ]

        # Negative documents (not relevant to the query)
        negative_documents = [
            "The weather forecast shows rain for the next three days.",
            "Python is a popular programming language for data science and machine learning.",
            "The capital of France is Paris, which is known for its historic architecture."
        ]
    else:

        # Combine all documents
        query = "Where, in the body, would the Cobb-Stainsby excision arthroplasty be performed?"
        positive_documents = [
            "TITLE: Effectiveness of the Cobb-Stainsby excision arthroplasty. BACKGROUND: Dislocated metatarsophalangeal joints from clawed or hammer toes can be a disabling consequence of several conditions. The Cobb-Stainsby forefoot arthroplasty combines partial phalangectomy (Stainsby) with extensor tendon transfer to the metatarsal head (Cobb). We present a retrospective, three surgeon case series of 215 toes in 126 patients.METHODS: Early results and complications were gathered from the medical charts of 126 patients who met the inclusion criteria. Seventy-five patients were contactable by phone with a follow up range of 12-82 months (median follow up 45 months). Primary outcome measures were improvement of pain and function, reduction in plantar callosities and cosmetic improvement of the deformity.RESULTS: Pre-operatively all patients presented with pain and shoe wear problems. Post-operatively seventy-two patients (96%) were satisfied, 72 (96%) reported pain relief, 55 (73%) were happy with toe control, 61 (81%) were pleased with cosmesis and 56 (75%) reported unlimited daily activities. Superficial wound infections were observed in 13 of the 126 patients (10%) and two in 75 patients (2%) developed recurrent clawing.CONCLUSION: Our case series demonstrates improved outcomes over alternatives such as the Weil's osteotomy."
        ]
        negative_documents = [
            "TITLE: Non-invasive fetal electrocardiography, electrohysterography and speckle-tracking echocardiography in the second trimester: study protocol of a longitudinal prospective cohort study (BEATS-study). BACKGROUND: Worldwide, hypertensive disorders of pregnancy (HDP), fetal growth restriction (FGR) and preterm birth remain the leading causes of maternal and fetal pregnancy-related mortality and (long-term) morbidity. Fetal cardiac deformation changes can be the first sign of placental dysfunction, which is associated with HDP, FGR and preterm birth. In addition, preterm birth is likely associated with changes in electrical activity across the uterine muscle. Therefore, fetal cardiac function and uterine activity can be used for the early detection of these complications in pregnancy. Fetal cardiac function and uterine activity can be assessed by two-dimensional speckle-tracking echocardiography (2D-STE), non-invasive fetal electrocardiography (NI-fECG), and electrohysterography (EHG). This study aims to generate reference values for 2D-STE, NI-fECG and EHG parameters during the second trimester of pregnancy and to investigate the diagnostic potential of these parameters in the early detection of HDP, FGR and preterm birth.METHODS: In this longitudinal prospective cohort study, eligible women will be recruited from a tertiary care hospital and a primary midwifery practice. In total, 594 initially healthy pregnant women with an uncomplicated singleton pregnancy will be included. Recordings of NI-fECG and EHG will be made weekly from 22 until 28 weeks of gestation and 2D-STE measurements will be performed 4-weekly at 16, 20, 24 and 28 weeks gestational age. Retrospectively, pregnancies complicated with pregnancy-related diseases will be excluded from the cohort. Reference values for 2D-STE, NI-fECG and EHG parameters will be assessed in uncomplicated pregnancies. After, 2D-STE, NI-fCG and EHG parameters measured during gestation in complicated pregnancies will be compared with these reference values.DISCUSSION: This will be the a large prospective study investigating new technologies that could potentially have a high impact on antepartum fetal monitoring.TRIAL REGISTRATION: Registered on 26 March 2020 in the Dutch Trial Register (NL8769) via https:\/\/www.trialregister.nl\/trials and registered on 21 October 2020 to the Central Committee on Research Involving Human Subjects (NL73607.015.20) via https:\/\/www.toetsingonline.nl\/to\/ccmo_search.nsf\/Searchform?OpenForm",
            "TITLE: CH3NH3PbBr3 Thin Film Served as Guided-Wave Layer for Enhancing the Angular Sensitivity of Plasmon Biosensor. CH3NH3PbBr3 perovskite thin film is used as a guided-wave layer and coated on the surface of an Au film to form the Au-perovskite hybrid structure. Using the hybrid structure, a perovskite-based guided-wave surface plasmon resonance (GWSPR) biosensor is proposed with high angular sensitivity. First, it is found that the electric field at the sensing interface is improved by the CH3NH3PbBr3 perovskite thin film, thereby enhancing the sensitivity. The result demonstrates that the angular sensitivity of the Au-perovskite-based GWSPR biosensor is as high as 278.5<B0>\/RIU, which is 110.2% higher than that of a conventional Au-based surface plasmon resonance (SPR) biosensor. Second, the selection of the coupling prism in the configuration of the GWSPR biosensor is also analyzed, and it indicates that a low refractive index (RI) prism can generate greater sensitivity. Therefore, the low-RI BK7 prism is served as the coupling prism for the proposed GWSPR biosensor. Finally, the proposed GWSPR sensing structure can not only be used for liquid sensing, but also for gas sensing, and it has also been demonstrated that the GWSPR gas sensor is 2.8 times more sensitive than the Au-based SPR gas sensor.",
            "TITLE: Sensing Techniques for Organochlorides through Intermolecular Interaction with Bicyclic Amidines. Toxic organochloride molecules are widely used in industry for various purposes. With their high volatility, the direct detection of organochlorides in environmental samples is challenging. Here, a new organochloride detection mechanism using 1,5-diazabicyclo[4.3.0]non-5-ene (DBN) is introduced to simplify a sensing method with higher detection sensitivity. Three types of organochloride compounds-trichloroethylene (TCE), dichloromethane (DCM), and dichlorodiphenyltrichloroethane (DDT)-were targeted to understand DCM conjugation chemistry by using nuclear magnetic resonance (NMR) and liquid chromatography with a mass spectrometer (LC-MS). 13C-NMR spectra and LC-MS data indicated that DBN can be labeled on these organochloride compounds by chlorine-nitrogen interaction. Furthermore, to demonstrate the organochloride sensing capability, the labeling yield and limit of detection were determined by a colorimetric assay as well as micellar electrokinetic chromatography (MEKC). The interaction with DBN was most appreciable for TCE, among other organochlorides. TCE was detected at picomolar levels, which is two orders of magnitude lower than the maximum contaminant level set by the United States Environmental Protection Agency. MEKC, in conjunction with this DBN-labeling method, enables us to develop a field-deployable sensing platform for detecting toxic organochlorides with high sensitivity.",
            "TITLE: Breast Mass Classification Using Diverse Contextual Information and Convolutional Neural Network. Masses are one of the early signs of breast cancer, and the survival rate of women suffering from breast cancer can be improved if masses can be correctly identified as benign or malignant. However, their classification is challenging due to the similarity in texture patterns of both types of mass. The existing methods for this problem have low sensitivity and specificity. Based on the hypothesis that diverse contextual information of a mass region forms a strong indicator for discriminating benign and malignant masses and the idea of the ensemble classifier, we introduce a computer-aided system for this problem. The system uses multiple regions of interest (ROIs) encompassing a mass region for modeling diverse contextual information, a single ResNet-50 model (or its density-specific modification) as a backbone for local decisions, and stacking with SVM as a base model to predict the final decision. A data augmentation technique is introduced for fine-tuning the backbone model. The system was thoroughly evaluated on the benchmark CBIS-DDSM dataset using its provided data split protocol, and it achieved a sensitivity of 98.48% and a specificity of 92.31%. Furthermore, it was found that the system gives higher performance if it is trained and tested using the data from a specific breast density BI-RADS class. The system does not need to fine-tune\/train multiple CNN models; it introduces diverse contextual information by multiple ROIs. The comparison shows that the method outperforms the state-of-the-art methods for classifying mass regions into benign and malignant. It will help radiologists reduce their burden and enhance their sensitivity in the prediction of malignant masses."
        ]

    all_documents = positive_documents + negative_documents

    print(f"\n📝 Query: {query}")
    print(f"\n✅ Positive Documents ({len(positive_documents)}):")
    for i, doc in enumerate(positive_documents, 1):
        print(f"  {i}. {doc}")
        print(f"    Tokens: {num_tokens(doc)}")

    print(f"\n❌ Negative Documents ({len(negative_documents)}):")
    for i, doc in enumerate(negative_documents, 1):
        print(f"  {i}. {doc}")
        print(f"    Tokens: {num_tokens(doc)}")

    # Compute similarities
    print("\n🔢 Computing similarities...")
    similarities = tester.compute_similarities(query, all_documents)

    # Display results
    print(f"\n📊 Similarity Scores:")
    print("-" * 60)
    for i, (doc, score) in enumerate(zip(all_documents, similarities)):
        doc_type = "POSITIVE" if i < len(positive_documents) else "NEGATIVE"
        print(f"{doc_type:8} | Score: {score:.4f} | {doc[:60]}...")

    # Rank all documents
    ranked_docs = tester.rank_documents(query, all_documents)

    print(f"\n🏆 Ranked Documents (by similarity):")
    print("-" * 70)
    for rank, (orig_idx, doc, score) in enumerate(ranked_docs, 1):
        doc_type = "POSITIVE" if orig_idx < len(
            positive_documents) else "NEGATIVE"
        print(f"Rank {rank}: {score:.4f} | {doc_type:8} | {doc[:50]}...")

    # Analyze results
    print(f"\n📈 Analysis:")
    pos_scores = similarities[:len(positive_documents)]
    neg_scores = similarities[len(positive_documents):]

    print(
        f"  • Average positive document similarity: {np.mean(pos_scores):.4f}")
    print(
        f"  • Average negative document similarity: {np.mean(neg_scores):.4f}")
    print(f"  • Best positive score: {np.max(pos_scores):.4f}")
    print(f"  • Best negative score: {np.max(neg_scores):.4f}")

    # Check if positive documents rank higher than negative ones
    top_k = 3
    top_k_indices = [orig_idx for orig_idx, _, _ in ranked_docs[:top_k]]
    positive_in_top_k = sum(1 for idx in top_k_indices
                            if idx < len(positive_documents))

    print(
        f"  • Positive documents in top-{top_k}: {positive_in_top_k}/{len(positive_documents)}"
    )

    if positive_in_top_k == len(positive_documents):
        print("  ✅ All positive documents ranked in top positions!")
    else:
        print("  ⚠️ Some negative documents ranked higher than positive ones.")

    ipdb.set_trace()
    pass


def interactive_similarity_tester():
    """
    Interactive function to test similarities with custom queries and documents.
    """
    print("\n🎮 Interactive Similarity Tester")
    print("=" * 40)

    tester = RetrievalTester()

    while True:
        print("\nEnter a query (or 'quit' to exit):")
        query = input("> ").strip()

        if query.lower() in ['quit', 'exit', 'q']:
            break

        if not query:
            continue

        print(
            "\nEnter documents separated by '|' (e.g., 'doc1 | doc2 | doc3'):")
        doc_input = input("> ").strip()

        if not doc_input:
            continue

        documents = [doc.strip() for doc in doc_input.split('|')]

        # Compute and display similarities
        similarities = tester.compute_similarities(query, documents)

        print(f"\n📊 Results for query: '{query}'")
        print("-" * 50)

        ranked_docs = tester.rank_documents(query, documents)
        for rank, (orig_idx, doc, score) in enumerate(ranked_docs, 1):
            print(f"Rank {rank}: {score:.4f} | {doc}")


def test_encoder_methods():
    """
    Test the individual encoder methods to show how they work.
    """
    print("\n🔧 Testing Individual Encoder Methods")
    print("=" * 50)

    tester = RetrievalTester()
    ipdb.set_trace()

    # Test query encoding
    queries = ["What causes diabetes?", "How to treat high blood pressure?"]
    print(f"\n📝 Encoding {len(queries)} queries:")
    for i, query in enumerate(queries):
        print(f"  {i+1}. {query}")

    query_embeddings = tester.encode_queries(queries)
    print(f"Query embeddings shape: {query_embeddings.shape}")
    print(f"Query embedding norms: {np.linalg.norm(query_embeddings, axis=1)}"
          )  # Should be ~1.0 (normalized)

    # Test document encoding
    documents = [
        "Diabetes is caused by high blood sugar levels.",
        "High blood pressure can be treated with medication and lifestyle changes.",
        "Machine learning is a subset of artificial intelligence."
    ]
    print(f"\n📄 Encoding {len(documents)} documents:")
    for i, doc in enumerate(documents):
        print(f"  {i+1}. {doc}")

    doc_embeddings = tester.encode_documents(documents)
    print(f"Document embeddings shape: {doc_embeddings.shape}")
    print(f"Document embedding norms: {np.linalg.norm(doc_embeddings, axis=1)}"
          )  # Should be ~1.0 (normalized)

    # Test similarity computation
    print(f"\n🔍 Computing similarities...")
    for i, query in enumerate(queries):
        query_emb = query_embeddings[i:i + 1]  # Shape: (1, embedding_dim)
        similarities = np.dot(query_emb,
                              doc_embeddings.T)[0]  # Shape: (n_docs,)

        print(f"\nQuery: {query}")
        for j, (doc, sim) in enumerate(zip(documents, similarities)):
            print(f"  Doc {j+1}: {sim:.4f} | {doc[:50]}...")

    ipdb.set_trace()
    pass


if __name__ == "__main__":
    # Run the main example test
    test_retrieval_example()

    # Test individual encoder methods
    # test_encoder_methods()

    # Uncomment the line below for interactive testing
    # interactive_similarity_tester()

    print("\n🎉 All tests completed!")
