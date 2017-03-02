package niukun;

import java.io.File;
import java.util.Collection;

import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.models.word2vec.Word2Vec;
import org.deeplearning4j.text.sentenceiterator.LineSentenceIterator;
import org.deeplearning4j.text.sentenceiterator.SentenceIterator;
import org.deeplearning4j.text.sentenceiterator.SentencePreProcessor;
import org.deeplearning4j.text.tokenization.tokenizer.preprocessor.CommonPreprocessor;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class BuildModel_01 {
	private static Logger log = LoggerFactory.getLogger(BuildModel_01.class);
	public static void main(String[] args) throws Exception {
		log.info("Load data....");
        SentenceIterator iter = new LineSentenceIterator(new File("C:/D/NLPIR/paper/files/merge/noNormalize_Segment/noNormalize_Segment.txt"));
        iter.setPreProcessor(new SentencePreProcessor() {
            public String preProcess(String sentence) {
                return sentence.toLowerCase();
            }
        });
        
        TokenizerFactory t = new DefaultTokenizerFactory();
        t.setTokenPreProcessor(new CommonPreprocessor());
        
        log.info("Building model....");
        Word2Vec vec = new Word2Vec.Builder()
                .minWordFrequency(20)
                .iterations(1)
                .layerSize(100)
                .seed(42)
                .windowSize(5)
                .iterate(iter)
                .tokenizerFactory(t)
                .build();

        log.info("Fitting Word2Vec model....");
        vec.fit();
        
        WordVectorSerializer.writeWordVectors(vec, "C:/D/NLPIR/paper/files/merge/noNormalize_Segment/vector.txt");

        log.info("Closest Words:");
        Collection<String> lst = vec.wordsNearest("北京", 10);
        System.out.println(lst);
//        UiServer server = UiServer.getInstance();
//        System.out.println("Started on port " + server.getPort());
        
	}
	
}
