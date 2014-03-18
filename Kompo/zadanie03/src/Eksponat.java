import java.util.Comparator;
import java.util.Date;

class Eksponat implements Comparator<Eksponat>, Comparable<Eksponat>{
	 public enum lokalizacja{EKSPOZYCJA, MAGAZYN, KONSERWACJA, WYPO¯YCZONY};
	
	 private String nazwa;
	 private int numer;
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
	 
	 public int compare(Eksponat obj1, Eksponat obj2)
	 {
		return 0; 
	 }
	 
	 public int compareTo(Eksponat obj1)
	 {
		return 0; 
	 }
	 
	 boolean equals(Eksponat obj1)
	 {
		 return true;
	 }
}
