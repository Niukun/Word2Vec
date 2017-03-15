package niukun;

import java.io.File;
import java.io.IOException;
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

import com.sun.xml.internal.messaging.saaj.soap.impl.ElementFactory;

public class BuildModel_01 {
	private static Logger log = LoggerFactory.getLogger(BuildModel_01.class);
	public static void main(String[] args) throws Exception {
//		getVector("C:/D/NLPIR/paper/files/merge/clean3.0/clean3.0.txt","C:/D/NLPIR/paper/files/merge/clean3.0/clean3.0_50_2.txt");
//		getVector("E:/NLPIR/merge/tensite/tensiteSegstopWordsRemoved.txt","E:/NLPIR/merge/tensite/tensiteVector_50_2.txt");
		getVector("E:/NLPIR/merge/sohusite_tensite/sohusite_tensiteSegstopWordsRemoved.txt","E:/NLPIR/merge/sohusite_tensite/sohusite_tensiteVector_50_2.txt");
//		getVector("E:/NLPIR/merge/sohusite/sohusiteSegstopWordsRemoved.txt","E:/NLPIR/merge/sohusite/sohusiteVector_50_2.txt");
        
	}
	public static void getVector(String srcFile,String vecFile) throws IOException {
		log.info("Load data....");
        SentenceIterator iter = new LineSentenceIterator(new File(srcFile));
        iter.setPreProcessor(new SentencePreProcessor() {
            public String preProcess(String sentence) {
                return sentence.toLowerCase();
            }
        });
        
        TokenizerFactory t = new DefaultTokenizerFactory();
        t.setTokenPreProcessor(new CommonPreprocessor());
        
        log.info("Building model....");
        Word2Vec vec = new Word2Vec.Builder()
                .minWordFrequency(5)
                .iterations(2)
                .layerSize(50)
                .seed(42)
                .windowSize(5)
                .iterate(iter)
                .tokenizerFactory(t)
                .build();

        log.info("Fitting Word2Vec model....");
        vec.fit();
        
        WordVectorSerializer.writeWordVectors(vec, vecFile);

        log.info(srcFile + "Finished.....");
        log.info("*********************************************************");
//        Collection<String> lst = vec.wordsNearest("北京", 10);
//        System.out.println(lst);
//        UiServer server = UiServer.getInstance();
//        System.out.println("Started on port " + server.getPort());
	}
	
}
