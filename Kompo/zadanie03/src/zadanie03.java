import java.sql.Date;


public class zadanie03 {

	/**
	 * @param args
	 */
	public static void main(String[] args) {
		// TODO Auto-generated method stub
		Date mojaData = Date.valueOf("2014-12-04");
		//enum lokalizacja{EKSPOZYCJA, MAGAZYN, KONSERWACJA, WYPOŻYCZONY};
		
		Eksponat mojEksponat = new Eksponat("Moj eksponat", 1, Eksponat.lokalizacja.MAGAZYN, mojaData);
		System.out.println(mojEksponat.toString());
	}

}
