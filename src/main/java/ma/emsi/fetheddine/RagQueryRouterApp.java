package ma.emsi.fetheddine;

import dev.langchain4j.data.document.Document;
import dev.langchain4j.data.document.DocumentParser;
import dev.langchain4j.data.document.DocumentSplitter;
import dev.langchain4j.data.document.loader.FileSystemDocumentLoader;
import dev.langchain4j.data.document.parser.apache.tika.ApacheTikaDocumentParser;
import dev.langchain4j.data.document.splitter.DocumentSplitters;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.memory.chat.MessageWindowChatMemory;
import dev.langchain4j.model.chat.ChatModel;
// Imports pour le modèle Google AI Embedding
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.model.googleai.GoogleAiEmbeddingModel;
import dev.langchain4j.model.googleai.GoogleAiGeminiChatModel;
import dev.langchain4j.model.input.PromptTemplate;
import dev.langchain4j.rag.DefaultRetrievalAugmentor;
import dev.langchain4j.rag.RetrievalAugmentor;
import dev.langchain4j.rag.content.retriever.ContentRetriever;
import dev.langchain4j.rag.content.retriever.EmbeddingStoreContentRetriever;
import dev.langchain4j.rag.query.Query;
import dev.langchain4j.rag.query.router.QueryRouter;
import dev.langchain4j.service.AiServices;
import dev.langchain4j.store.embedding.EmbeddingStore;
// Import pour EmbeddingStoreIngestor
import dev.langchain4j.store.embedding.EmbeddingStoreIngestor;
import dev.langchain4j.store.embedding.inmemory.InMemoryEmbeddingStore;

import java.net.URI;
import java.net.URISyntaxException;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Collection;
import java.util.Collections;
import java.util.Map;
import java.util.Scanner;
import java.util.logging.ConsoleHandler;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 * Main application class to demonstrate a RAG system with selective retrieval.
 * This program implements a custom QueryRouter that decides whether to fetch
 * context from a document store based on the user's query topic.
 *
 * @author Fetheddine Aya
 * @version 1.3
 */
public class RagQueryRouterApp {

    // Helper utility to resolve resource file paths.
    private static Path getPath(String fileName) {
        try {
            URI fileUri = RagQueryRouterApp.class.getClassLoader().getResource(fileName).toURI();
            return Paths.get(fileUri);
        } catch (URISyntaxException e) {
            throw new RuntimeException("Failed to resolve resource path: " + fileName, e);
        }
    }

    // Configures the global logger for the langchain4j package to FINE level.
    private static void configureLogger() {
        Logger packageLogger = Logger.getLogger("dev.langchain4j");
        packageLogger.setLevel(Level.FINE);
        ConsoleHandler handler = new ConsoleHandler();
        handler.setLevel(Level.FINE);
        packageLogger.addHandler(handler);
    }

    /**
     * Ingestion pipeline: Loads, parses, splits, and embeds a single document
     * using the EmbeddingStoreIngestor.
     *
     * @param resourceName   The name of the document in the resources folder.
     * @param embeddingModel The model to use for embedding text segments.
     * @return A populated in-memory EmbeddingStore.
     */
    private static EmbeddingStore<TextSegment> ingestDocument(String resourceName, EmbeddingModel embeddingModel) {
        System.out.println("Ingestion de '" + resourceName + "' en cours...");
        Path documentPath = getPath(resourceName);

        DocumentParser parser = new ApacheTikaDocumentParser();
        Document document = FileSystemDocumentLoader.loadDocument(documentPath, parser);

        DocumentSplitter splitter = DocumentSplitters.recursive(300, 30);
        EmbeddingStore<TextSegment> embeddingStore = new InMemoryEmbeddingStore<>();

        // Utilisation de EmbeddingStoreIngestor (cohérent avec TestRoutage.java)
        EmbeddingStoreIngestor ingestor = EmbeddingStoreIngestor.builder()
                .documentSplitter(splitter)
                .embeddingModel(embeddingModel)
                .embeddingStore(embeddingStore)
                .build();

        ingestor.ingest(document);

        System.out.println("Ingestion terminée.");
        return embeddingStore;
    }

    /**
     * Inner class implementing the custom QueryRouter.
     */
    static class TopicBasedQueryRouter implements QueryRouter {

        private final ChatModel chatModel;
        private final ContentRetriever documentRetriever;

        // Prompt Template en français (cohérent avec TestPasDeRag.java)
        private final PromptTemplate routingPromptTemplate = PromptTemplate.from(
                "Est-ce que la requête '{{query}}' porte sur l'IA (Intelligence Artificielle) ou le 'RAG' (Retrieval Augmented Generation) ? "
                        + "Réponds seulement par 'oui' ou 'non'."
        );

        public TopicBasedQueryRouter(ChatModel chatModel, ContentRetriever documentRetriever) {
            this.chatModel = chatModel;
            this.documentRetriever = documentRetriever;
        }

        @Override
        public Collection<ContentRetriever> route(Query query) {
            String prompt = routingPromptTemplate.apply(Map.of("query", query.text())).text();
            String decision = chatModel.chat(prompt);

            // Logs en français (cohérent avec TestPasDeRag.java)
            if (decision.toLowerCase().trim().contains("oui")) {
                System.out.println("Routage : [RAG] activé. (Réponse LLM: " + decision + ")");
                return Collections.singletonList(documentRetriever);
            } else {
                System.out.println("Routage : Pas de RAG activé. (Réponse LLM: " + decision + ")");
                return Collections.emptyList();
            }
        }
    }

    public static void main(String[] args) {
        configureLogger();

        String llmKey = System.getenv("GEMINI_KEY");
        if (llmKey == null || llmKey.isEmpty()) {
            System.err.println("GEMINI_KEY environment variable is not set.");
            return;
        }

        // --- PHASE 1: MODEL INITIALIZATION ---
        ChatModel chatLlm = GoogleAiGeminiChatModel.builder()
                .apiKey(llmKey)
                .modelName("gemini-2.5-flash")
                .temperature(0.3)
                .logRequests(true)
                .logResponses(true)
                .build();

        // **Modèle d'embedding "revert" vers GoogleAiEmbeddingModel**
        EmbeddingModel docEmbeddingModel = GoogleAiEmbeddingModel.builder()
                .apiKey(llmKey)
                .modelName("text-embedding-004")
                .build();

        // --- PHASE 2: INGESTION ---
        EmbeddingStore<TextSegment> ragDocumentStore = ingestDocument("rag.pdf", docEmbeddingModel);

        // --- PHASE 3: RAG PIPELINE SETUP ---
        ContentRetriever documentContentRetriever = EmbeddingStoreContentRetriever.builder()
                .embeddingStore(ragDocumentStore)
                .embeddingModel(docEmbeddingModel)
                .maxResults(2)
                .build();

        QueryRouter topicRouter = new TopicBasedQueryRouter(chatLlm, documentContentRetriever);

        RetrievalAugmentor ragAugmentor = DefaultRetrievalAugmentor.builder()
                .queryRouter(topicRouter)
                .build();

        Assistant assistant = AiServices.builder(Assistant.class)
                .chatModel(chatLlm)
                .chatMemory(MessageWindowChatMemory.withMaxMessages(10))
                .retrievalAugmentor(ragAugmentor)
                .build();


        System.out.println("\nBonjour ! Je réponds aux questions sur le RAG (et ignore le reste).");
        Scanner scanner = new Scanner(System.in);
        while (true) {
            System.out.print("\nVous : ");
            String question = scanner.nextLine();

            if (question.equalsIgnoreCase("stop")) {
                break;
            }

            String reponse = assistant.chat(question);
            System.out.println("Assistant : " + reponse);
        }
        scanner.close();
    }
}