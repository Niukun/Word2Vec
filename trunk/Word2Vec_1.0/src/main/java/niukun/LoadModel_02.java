package niukun;

import java.util.Collection;

import org.deeplearning4j.models.embeddings.WeightLookupTable;
import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.models.word2vec.Word2Vec;

public class LoadModel_02 {

	public static void main(String[] args) {
		Word2Vec word2Vec = WordVectorSerializer.readWord2VecModel("C:/D/NLPIR/paper/files/merge/noNormalize_Segment/vector.txt");
		WeightLookupTable weightLookupTable = word2Vec.lookupTable();


		Collection<String> lst = word2Vec.wordsNearest("男人", 10);
        System.out.println(lst);
        lst = word2Vec.wordsNearest("乾隆", 10);
        System.out.println(lst);
        lst = word2Vec.wordsNearest("习近平", 10);
        System.out.println(lst);
        lst = word2Vec.wordsNearest("改革", 10);
        System.out.println(lst);
        lst = word2Vec.wordsNearest("枪", 10);
        System.out.println(lst);
        lst = word2Vec.wordsNearest("阅读", 10);
        System.out.println(lst);
		
	}

}
