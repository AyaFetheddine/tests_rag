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
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.model.googleai.GoogleAiEmbeddingModel;
import dev.langchain4j.model.googleai.GoogleAiGeminiChatModel;
import dev.langchain4j.rag.DefaultRetrievalAugmentor;
import dev.langchain4j.rag.RetrievalAugmentor;
import dev.langchain4j.rag.content.retriever.ContentRetriever;
import dev.langchain4j.rag.content.retriever.EmbeddingStoreContentRetriever;
import dev.langchain4j.rag.query.router.LanguageModelQueryRouter;
import dev.langchain4j.rag.query.router.QueryRouter;
import dev.langchain4j.service.AiServices;
import dev.langchain4j.store.embedding.EmbeddingStore;
import dev.langchain4j.store.embedding.EmbeddingStoreIngestor;
import dev.langchain4j.store.embedding.inmemory.InMemoryEmbeddingStore;

import java.io.File;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Scanner;
import java.util.logging.ConsoleHandler;
import java.util.logging.Level;
import java.util.logging.Logger;

public class TestRoutage {
    private static void configureLogger() {
        Logger packageLogger = Logger.getLogger("dev.langchain4j");
        packageLogger.setLevel(Level.FINE);
        ConsoleHandler handler = new ConsoleHandler();
        handler.setLevel(Level.FINE);
        packageLogger.addHandler(handler);
    }
    // Method to create and ingest an embedding store from a document
    private static EmbeddingStore<TextSegment> createAndIngestEmbeddingStore(String documentPath, EmbeddingModel embeddingModel) {
        Path path = Paths.get(documentPath);
        DocumentParser parser = new ApacheTikaDocumentParser();
        Document document = FileSystemDocumentLoader.loadDocument(path, parser);

        DocumentSplitter splitter = DocumentSplitters.recursive(300, 0);
        EmbeddingStore<TextSegment> embeddingStore = new InMemoryEmbeddingStore<>();
        EmbeddingStoreIngestor ingestor = EmbeddingStoreIngestor.builder()
                .documentSplitter(splitter)
                .embeddingModel(embeddingModel)
                .embeddingStore(embeddingStore)
                .build();
        ingestor.ingest(document);
        return embeddingStore;
    }

    public static void main(String[] args) {
        configureLogger();
        // Phase 1: Ingestion
        EmbeddingModel embeddingModel = GoogleAiEmbeddingModel.builder()
                .apiKey(System.getenv("GEMINI_KEY"))
                .modelName("text-embedding-004")
                .build();

        EmbeddingStore<TextSegment> ragEmbeddingStore = createAndIngestEmbeddingStore("src/main/resources/rag.pdf", embeddingModel);
        EmbeddingStore<TextSegment> threatReportEmbeddingStore = createAndIngestEmbeddingStore("src/main/resources/threat_report.pdf", embeddingModel);

        // Phase 2: Retrieval
        ChatModel chatModel = GoogleAiGeminiChatModel.builder()
                .apiKey(System.getenv("GEMINI_KEY"))
                .modelName("gemini-2.5-flash")
                .temperature(0.3)
                .logRequestsAndResponses(true)
                .build();

        ContentRetriever ragContentRetriever = EmbeddingStoreContentRetriever.builder()
                .embeddingStore(ragEmbeddingStore)
                .embeddingModel(embeddingModel)
                .maxResults(2)
                .minScore(0.6)
                .build();

        ContentRetriever threatReportContentRetriever = EmbeddingStoreContentRetriever.builder()
                .embeddingStore(threatReportEmbeddingStore)
                .embeddingModel(embeddingModel)
                .maxResults(2)
                .minScore(0.6)
                .build();

        Map<ContentRetriever, String> retrieverMap = new HashMap<>();
        retrieverMap.put(ragContentRetriever, "Answers questions about Retrieval-Augmented Generation (RAG)");
        retrieverMap.put(threatReportContentRetriever, "Answers questions about a threat report");

        QueryRouter queryRouter = new LanguageModelQueryRouter(chatModel, retrieverMap);

        RetrievalAugmentor retrievalAugmentor = DefaultRetrievalAugmentor.builder()
                .queryRouter(queryRouter)
                .build();

        // Phase 3: Assistant Creation
        Assistant assistant = AiServices.builder(Assistant.class)
                .chatModel(chatModel)
                .retrievalAugmentor(retrievalAugmentor)
                .chatMemory(MessageWindowChatMemory.withMaxMessages(10))
                .build();

        // Test
        Scanner scanner = new Scanner(System.in);
        while (true) {
            System.out.print("You: ");
            String query = scanner.nextLine();
            if (query.equalsIgnoreCase("exit")) {
                break;
            }
            String response = assistant.chat(query);
            System.out.println("Assistant: " + response);
        }
        scanner.close();
    }
}
