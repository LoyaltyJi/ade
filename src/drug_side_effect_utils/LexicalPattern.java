package drug_side_effect_utils;

import cn.fox.nlp.EnglishPos;
import cn.fox.nlp.Punctuation;
import cn.fox.utils.CharCode;

public class LexicalPattern {
	public int ctUpCase;
	public int ctAlpha;
	public int ctNum;
	/*public int ctHyphen;
	public int ctBracket;
	public int ctApostrophe;
	public int ctComma;*/
	public int ctPunc;
	
	public void clear() {
		ctUpCase = 0;
		ctAlpha = 0;
		ctNum = 0;
		/*ctHyphen = 0;
		ctBracket = 0;
		ctApostrophe = 0;
		ctComma = 0;*/
		ctPunc = 0;
	}
	
	public void getAll(String s) {
		char[] chs = s.toCharArray();
		clear();

		for(int i=0;i<chs.length;i++) {
			if(CharCode.isUpperCase(chs[i]))
				ctUpCase++;	
			if(CharCode.isEnAlpha(chs[i]))
				ctAlpha++;
			if(CharCode.isNumber(chs[i]))
				ctNum++;
			/*if(chs[i] == '-')
				ctHyphen++;	
			if(chs[i] == '(' || chs[i] == ')' || chs[i] == '[' || chs[i] == ']' || chs[i] == '{' || chs[i] == '}')
				ctBracket++;
			if(chs[i] == '\'')
				ctApostrophe++;	
			if(chs[i] == ',')
				ctComma++;	*/
			if(Punctuation.isEnglishPunc(chs[i]))
				ctPunc++;
		}
		
	}
	
	// get the number of uppercase characters
	public static int getUpCaseNum(String s) {
		char[] chs = s.toCharArray();

		int count = 0;
		for(int i=0;i<chs.length;i++) {
			if(CharCode.isUpperCase(chs[i]))
				count++;	
		}
		return count;
	}
	
	public static int getAlphaNum(String s) {
		char[] chs = s.toCharArray();

		int count = 0;
		for(int i=0;i<chs.length;i++) {
			if(CharCode.isEnAlpha(chs[i]))
				count++;	
		}
		return count;
	}
	
	// get the number of number characters
	public static int getNumNum(String s) {
		char[] chs = s.toCharArray();

		int countNum = 0;
		for(int i=0;i<chs.length;i++) {
			if(CharCode.isNumber(chs[i]))
				countNum++;
		}
		return countNum;
	}
	
	// get the number of hyphen characters
	public static int getHyphenNum(String s) {
		char[] chs = s.toCharArray();

		int count = 0;
		for(int i=0;i<chs.length;i++) {
			if(chs[i] == '-')
				count++;	
		}
		return count;
	}
	// get the number of bracket characters
	public static int getBracketNum(String s) {
		char[] chs = s.toCharArray();

		int count = 0;
		for(int i=0;i<chs.length;i++) {
			if(chs[i] == '(' || chs[i] == ')' || chs[i] == '[' || chs[i] == ']' || chs[i] == '{' || chs[i] == '}')
				count++;	
		}
		return count;
	}
	
	public static int getApostropheNum(String s) {
		char[] chs = s.toCharArray();

		int count = 0;
		for(int i=0;i<chs.length;i++) {
			if(chs[i] == '\'')
				count++;	
		}
		return count;
	}
	
	public static int getCommaNum(String s) {
		char[] chs = s.toCharArray();

		int count = 0;
		for(int i=0;i<chs.length;i++) {
			if(chs[i] == ',')
				count++;	
		}
		return count;
	}
}
