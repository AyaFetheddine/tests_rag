package ma.emsi.fetheddine;

import dev.langchain4j.data.document.Document;
import dev.langchain4j.data.document.DocumentParser;
import dev.langchain4j.data.document.DocumentSplitter;
import dev.langchain4j.data.document.loader.FileSystemDocumentLoader;
import dev.langchain4j.data.document.parser.apache.tika.ApacheTikaDocumentParser;
import dev.langchain4j.data.document.splitter.DocumentSplitters;
import dev.langchain4j.data.embedding.Embedding;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.memory.chat.MessageWindowChatMemory;
import dev.langchain4j.model.chat.ChatModel;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.model.googleai.GoogleAiEmbeddingModel;
import dev.langchain4j.model.googleai.GoogleAiGeminiChatModel;
import dev.langchain4j.model.output.Response;
import dev.langchain4j.rag.DefaultRetrievalAugmentor;
import dev.langchain4j.rag.RetrievalAugmentor;
import dev.langchain4j.rag.content.retriever.ContentRetriever;
import dev.langchain4j.rag.content.retriever.EmbeddingStoreContentRetriever;
import dev.langchain4j.rag.content.retriever.WebSearchContentRetriever; // NOUVEL IMPORT
import dev.langchain4j.rag.query.router.DefaultQueryRouter; // NOUVEL IMPORT
import dev.langchain4j.rag.query.router.QueryRouter; // NOUVEL IMPORT
import dev.langchain4j.service.AiServices;
import dev.langchain4j.store.embedding.EmbeddingStore;
import dev.langchain4j.store.embedding.inmemory.InMemoryEmbeddingStore;
import dev.langchain4j.web.search.WebSearchEngine; // NOUVEL IMPORT
import dev.langchain4j.web.search.tavily.TavilyWebSearchEngine; // NOUVEL IMPORT

import java.net.URI;
import java.net.URISyntaxException;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.List;
import java.util.Scanner;
import java.util.logging.ConsoleHandler;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 * Test 5: RAG avec récupération Web.
 * Combine la récupération de documents locaux (PDF) avec une recherche Web (Tavily).
 * Utilise un DefaultQueryRouter pour agréger les résultats des deux sources.
 */
public class Test5WebRag {

    // Configure le logger de LangChain4j
    private static void configureLogger() {
        Logger packageLogger = Logger.getLogger("dev.langchain4j");
        packageLogger.setLevel(Level.FINE);
        ConsoleHandler handler = new ConsoleHandler();
        handler.setLevel(Level.FINE);
        packageLogger.addHandler(handler);
    }

    // Helper pour charger les ressources
    private static Path getPath(String fileName) {
        try {
            URI fileUri = Test5WebRag.class.getClassLoader().getResource(fileName).toURI();
            return Paths.get(fileUri);
        } catch (URISyntaxException e) {
            throw new RuntimeException(e);
        }
    }

    public static void main(String[] args) {
        configureLogger();

        // Récupération des clés API
        String geminiKey = System.getenv("GEMINI_KEY");
        String tavilyKey = System.getenv("TAVILY_KEY");

        if (geminiKey == null || geminiKey.isEmpty() || tavilyKey == null || tavilyKey.isEmpty()) {
            System.err.println("Erreur : Assurez-vous que les variables d'environnement GEMINI_KEY et TAVILY_KEY sont définies.");
            return;
        }

        // --- Configuration des modèles (comme TestRagNaif) ---
        ChatModel model = GoogleAiGeminiChatModel.builder()
                .apiKey(geminiKey)
                .modelName("gemini-2.5-flash")
                .temperature(0.3)
                .logRequests(true) // Modifié pour correspondre à RagQueryRouterApp
                .logResponses(true)
                .build();

        EmbeddingModel embeddingModel = GoogleAiEmbeddingModel.builder()
                .apiKey(geminiKey)
                .modelName("text-embedding-004")
                .build();

        // --- RETRIEVER 1 : RAG LOCAL (identique à TestRagNaif) ---
        Path documentPath = getPath("rag.pdf");
        DocumentParser parser = new ApacheTikaDocumentParser();
        Document document = FileSystemDocumentLoader.loadDocument(documentPath, parser);
        DocumentSplitter splitter = DocumentSplitters.recursive(300, 30);
        List<TextSegment> segments = splitter.split(document);

        EmbeddingStore<TextSegment> embeddingStore = new InMemoryEmbeddingStore<>();
        Response<List<Embedding>> embeddingsResponse = embeddingModel.embedAll(segments);
        embeddingStore.addAll(embeddingsResponse.content(), segments);

        ContentRetriever localContentRetriever = EmbeddingStoreContentRetriever.builder()
                .embeddingStore(embeddingStore)
                .embeddingModel(embeddingModel)
                .maxResults(2)
                .minScore(0.5)
                .build();

        System.out.println("Retriever local (PDF) initialisé.");

        // --- RETRIEVER 2 : RAG WEB (NOUVEAU) ---

        // 1. Créer le moteur de recherche (Tavily)
        WebSearchEngine webSearchEngine = TavilyWebSearchEngine.builder()
                .apiKey(tavilyKey)
                .build();

        // 2. Créer le ContentRetriever pour le Web
        ContentRetriever webContentRetriever = WebSearchContentRetriever.builder()
                .webSearchEngine(webSearchEngine)
                .build();

        System.out.println("Retriever web (Tavily) initialisé.");

        // --- NOUVELLE CONFIGURATION : ROUTEUR + AUGMENTOR ---

        // 3. Créer le QueryRouter avec les DEUX retrievers
        QueryRouter queryRouter = new DefaultQueryRouter(
                localContentRetriever,
                webContentRetriever
        );

        // 4. Créer le RetrievalAugmentor avec ce routeur
        RetrievalAugmentor retrievalAugmentor = DefaultRetrievalAugmentor.builder()
                .queryRouter(queryRouter)
                .build();

        // 5. Créer l'assistant avec le RetrievalAugmentor
        Assistant assistant = AiServices.builder(Assistant.class)
                .chatModel(model)
                .chatMemory(MessageWindowChatMemory.withMaxMessages(10))
                .retrievalAugmentor(retrievalAugmentor) // Utilise l'augmentor au lieu du retriever simple
                .build();

        // Lancer la conversation
        conversationAvec(assistant);
    }

    // Helper pour la boucle de conversation (identique à TestRagNaif)
    private static void conversationAvec(Assistant assistant) {
        try (Scanner scanner = new Scanner(System.in)) {
            while (true) {
                System.out.println("==================================================");
                System.out.println("Posez votre question (ou 'fin' pour quitter) : ");
                String question = scanner.nextLine();

                if (question.isBlank()) {
                    continue;
                }

                if ("fin".equalsIgnoreCase(question)) {
                    System.out.println("Conversation terminée.");
                    break;
                }

                System.out.println("==================================================");
                String reponse = assistant.chat(question);
                System.out.println("Assistant : " + reponse);
                System.out.println("==================================================");
            }
        }
    }
}