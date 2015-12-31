package biocorpus;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.util.List;

import javax.xml.parsers.DocumentBuilder;
import javax.xml.parsers.DocumentBuilderFactory;

import org.w3c.dom.Document;

import edu.stanford.nlp.ling.CoreLabel;

public class BioNlplab {

	public static void main(String[] args) throws Exception {
		OutputStreamWriter osw = new OutputStreamWriter(
				new FileOutputStream("F:/biomedical resource/bionlplaborg/1-grams-decode.txt"), "utf-8");
		
		BufferedReader br = null;
		String thisLine = null;
		
		br = new BufferedReader(new InputStreamReader(
				new FileInputStream("F:/biomedical resource/bionlplaborg/1-grams.txt"), "utf-8"));
		thisLine = null;
		int count = 0;
		while ((thisLine = br.readLine()) != null) {
			if(!thisLine.isEmpty()) {
				String[] splitted = thisLine.split("\t");
				osw.write(splitted[1].toLowerCase()+" ");
				count++;
				if(splitted[1].equals(".") || count%30==0)
					osw.write("\n");
			} 
		}
		br.close();

	}

}
