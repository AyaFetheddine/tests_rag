package ma.emsi.fetheddine;

import dev.langchain4j.data.document.Document;
import dev.langchain4j.data.document.DocumentParser;
import dev.langchain4j.data.document.DocumentSplitter;
import dev.langchain4j.data.document.loader.FileSystemDocumentLoader;
import dev.langchain4j.data.document.parser.apache.tika.ApacheTikaDocumentParser;
import dev.langchain4j.data.document.splitter.DocumentSplitters;
import dev.langchain4j.data.embedding.Embedding;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.memory.ChatMemory;
import dev.langchain4j.memory.chat.MessageWindowChatMemory;
import dev.langchain4j.model.chat.ChatModel;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.model.googleai.GoogleAiEmbeddingModel;
import dev.langchain4j.model.googleai.GoogleAiGeminiChatModel;
import dev.langchain4j.model.output.Response;
import dev.langchain4j.rag.content.retriever.ContentRetriever;
import dev.langchain4j.rag.content.retriever.EmbeddingStoreContentRetriever;
import dev.langchain4j.service.AiServices;
import dev.langchain4j.store.embedding.EmbeddingStore;
import dev.langchain4j.store.embedding.inmemory.InMemoryEmbeddingStore;

import java.net.URI;
import java.net.URISyntaxException;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.List;

public class TestRagNaif {

    // Helper pour charger les ressources
    private static Path getPath(String fileName) {
        try {
            URI fileUri = TestRagNaif.class.getClassLoader().getResource(fileName).toURI();
            return Paths.get(fileUri);
        } catch (URISyntaxException e) {
            throw new RuntimeException(e);
        }
    }

    public static void main(String[] args) {
        ChatModel model = GoogleAiGeminiChatModel.builder()
                .apiKey(System.getenv("GEMINI_KEY"))
                .modelName("gemini-2.5-flash")
                .temperature(0.3)
                .build();

        System.out.println("Phase 1 : Démarrage de l'ingestion...");
        Path documentPath = getPath("rag.pdf");
        DocumentParser parser = new ApacheTikaDocumentParser();
        Document document = FileSystemDocumentLoader.loadDocument(documentPath, parser);
        DocumentSplitter splitter = DocumentSplitters.recursive(300, 30);
        List<TextSegment> segments = splitter.split(document);
        EmbeddingModel embeddingModel = GoogleAiEmbeddingModel.builder()
                .apiKey(System.getenv("GEMINI_KEY"))
                .modelName("text-embedding-004")
                .build();
        Response<List<Embedding>> embeddingsResponse = embeddingModel.embedAll(segments);
        List<Embedding> embeddings = embeddingsResponse.content();
        EmbeddingStore<TextSegment> embeddingStore = new InMemoryEmbeddingStore<>();
        embeddingStore.addAll(embeddings, segments);
        System.out.println("Phase 1 : Ingestion terminée. " + segments.size() + " segments ajoutés.");

        ContentRetriever contentRetriever = EmbeddingStoreContentRetriever.builder()
                .embeddingStore(embeddingStore)
                .embeddingModel(embeddingModel)
                .maxResults(2)
                .minScore(0.5)
                .build();

        ChatMemory chatMemory = MessageWindowChatMemory.withMaxMessages(10);

        Assistant assistant = AiServices.builder(Assistant.class)
                .chatModel(model)
                .chatMemory(chatMemory)
                .contentRetriever(contentRetriever)
                .build();

        System.out.println("\nPhase 2 : Test avec une question unique.");
        String question = "Quelle est la signification de 'RAG' ; à quoi ça sert ?";

        System.out.println("Question : " + question);
        String reponse = assistant.chat(question);
        System.out.println("Assistant : " + reponse);

        System.out.println("\nProgramme terminé.");
    }
}