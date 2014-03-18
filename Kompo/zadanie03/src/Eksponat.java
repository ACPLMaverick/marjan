import java.util.Date;

class Eksponat {
	 private String nazwa;
	 private int numer;
	 public enum lokalizacja{EKSPOZYCJA, MAGAZYN, KONSERWACJA, WYPO¯YCZONY};
	 private lokalizacja l;
	 private Date data_zmiany;

	 public Eksponat(String n, int num, lokalizacja lok, Date data)
	 {
		 
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
}
