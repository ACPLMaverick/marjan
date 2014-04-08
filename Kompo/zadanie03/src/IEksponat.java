import java.util.Date;


public interface IEksponat {
	public void put(String wartosc);
	 public void put(int wartosc);
	 public void put(Date wartosc);
	 public String getNazwa();
	 public int getNumer();
	 public Date getData();
	 public boolean isNazwa(String naz);
	 public boolean isNumer(int num);
	 public boolean isData(Date d);
}
