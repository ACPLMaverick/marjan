import java.util.Comparator;
import java.util.Date;

class Eksponat implements Comparable{	
	 private String nazwa;
	 private int numer;
	 public enum lokalizacja{EKSPOZYCJA, MAGAZYN, KONSERWACJA, WYPO¯YCZONY};
	 private lokalizacja l;
	 private Date data_zmiany;

	 public Eksponat(String n, int num, lokalizacja lok, Date data)
	 {
		 this.nazwa = n;
		 this.numer = num;
		 this.data_zmiany = data;
		 this.l = lok;
	 }
	 
	 public void put(String wartosc)
	 {
		 nazwa = wartosc;
	 }
	 public void put(int wartosc)
	 {
		 numer = wartosc;
	 }
	 public void put(lokalizacja wartosc)
	 {
		 this.l = wartosc;
	 }
	 public void put(Date wartosc)
	 {
		 data_zmiany = wartosc;
	 }
	 public String setNazwa()
	 {
		 return nazwa;
	 }
	 public int setNumer()
	 {
		 return numer;
	 }
	 public lokalizacja setLokalizacja()
	 {
		 return l;
	 }
	 public Date setData()
	 {
		 return data_zmiany;
	 }
	 public boolean isNazwa(String naz)
	 {
		 if(nazwa == naz) return true;
		 else return false;
	 }
	 public boolean isNumer(int num)
	 {
		 if(numer == num) return true;
		 else return false;
	 }
	 public boolean isLokalizacja(lokalizacja lok)
	 {
		 if(l == lok) return true;
		 else return false;
	 }
	 public boolean isData(Date d)
	 {
		 if(data_zmiany == d) return true;
		 else return false;
	 }
	 
	 public String toString()
	 {
		 String myName = nazwa;
		 String myNumer = Integer.toString(numer);
		 String myDate = data_zmiany.toString();
		 String myLokalizacja = l.toString();
		 String myString = "";
		 myString = 
				 "NAZWA: " + myName + 
				 "\nNUMER: " + myNumer +
				 "\nLOKALIZACJA: " + myLokalizacja +
				 "\nDATA: " + myDate;
		 return myString;
	 }
	 
	 @Override
	 public int compareTo(Object o)
	 {
		int number = ((Eksponat) o).setNumer();
		return this.numer - number; 
	 }
	 
	 
	 public static Comparator<Eksponat> EksponatNameComparator = new Comparator<Eksponat>(){
		
		public int compare(Eksponat o1, Eksponat o2) {
			String eksponatNazwa1 = o1.setNazwa().toUpperCase();
			String eksponatNazwa2 = o2.setNazwa().toUpperCase();
			return eksponatNazwa1.compareTo(eksponatNazwa2);
		}
	 };
	 
	 public static Comparator<Eksponat> EksponatLocComparator = new Comparator<Eksponat>(){
		 public int compare(Eksponat o1, Eksponat o2) {
			 lokalizacja l1 = o1.setLokalizacja();
			 lokalizacja l2 = o2.setLokalizacja();
			 return l1.compareTo(l2);
		 }
	 };
	 
	 public static Comparator<Eksponat> EksponatDateComparator = new Comparator<Eksponat>(){
		 public int compare(Eksponat o1, Eksponat o2) {
			 Date d1 = o1.setData();
			 Date d2 = o2.setData();
			 return d1.compareTo(d2);
		 }
	 };

	/* @Override
	 public boolean equals(Object obj1)
	 {
		 if(obj1 == this) {return true;}
		 if(obj1 == null || obj1.getClass() != this.getClass()) {return false;}	//aby unikn¹æ ClassCastException
		 Eksponat eks = (Eksponat) obj1;
		 return numer = eks.numer && (nazwa = eks.nazwa || (nazwa != null && ))
	 }*/
}
