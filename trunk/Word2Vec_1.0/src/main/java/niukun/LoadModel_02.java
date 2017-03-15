package niukun;

import java.util.Collection;
import java.util.List;
import java.util.Set;
import java.util.TreeSet;

import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.models.word2vec.Word2Vec;

public class LoadModel_02 {
	private static Word2Vec word2Vec;
	private static Set<String> set = new TreeSet<String>();
	private static List<String> list;

	public static void main(String[] args) {
		 Word2Vec word2Vec = WordVectorSerializer.readWord2VecModel("G:/学习工具/计算机/NLP/word2vec/bin/GoogleNews-vectors-negative300.bin.gz");
		 Collection<String> lst = word2Vec.wordsNearest("day", 10);
		 for (String string : lst) {
			System.out.println(string + "\t" + word2Vec.similarity(string, "day"));
		 }
//		word2Vec = WordVectorSerializer
//				.readWord2VecModel("C:/D/NLPIR/paper/files/merge/sohusite_tensite/sohusite_tensiteVector_100_1.txt");
		 
//		 System.out.println(word2Vec.similarity("枪支", "弹药"));
		
	}

}
