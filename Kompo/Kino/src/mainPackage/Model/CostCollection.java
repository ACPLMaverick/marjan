package mainPackage.Model;

import java.sql.Time;
import java.util.ArrayList;
import java.util.Date;

// TODO: Auto-generated Javadoc
/**
 * Klasa przechowuje wydatki bπdü przychody, ktÛre uøytkownik kupi bπdü zarezerwuje, zaleønie od uøycia.
 */
public class CostCollection {
	
	private ArrayList<Cost> costs;
	
	/**
	 * Tworzy nowy obiekt typu CostCollection.
	 */
	public CostCollection()
	{
		costs = new ArrayList<Cost>();
	}
	
	/**
	 * Dodaje koszt do kolekcji.
	 *
	 * @param cost koszt
	 */
	public void add(Cost cost)
	{
		costs.add(cost);
	}
	
	/**
	 * Usuwa element w kolekcji pod wskazanym indeksem.
	 *
	 * @param i indeks w kolekcji.
	 */
	public void delete(int i)
	{
		costs.remove(i);
	}
	
	/**
	 * Zwraca koszt z kolekcji, pod wskazanym indeksem.
	 *
	 * @param i indeks w kolekcji.
	 * @return koszt pod wskazanym indeksem.
	 */
	public Cost get(int i)
	{
		return costs.get(i);
	}
	
	/**
	 * Zwraca kolekcje kosztow.
	 *
	 * @return kolekcje kosztow.
	 */
	public ArrayList<Cost> get()
	{
		return costs;
	}
	
	/**
	 * Zwraca sume wszystkich kosztow z przedzialu miedzy minimalna i maksymalna daty.
	 *
	 * @param dateMin minimalna data.
	 * @param dateMax maksymalna data.
	 * @return sume wszystkich kosztow z zadanego przedzialu.
	 */
	public double getWholeCosts(Date dateMin, Date dateMax)
	{
		if(dateMin == null)
		{
			dateMin = new Date();
			dateMin.setTime(Long.MIN_VALUE);
		}
		
		if(dateMax == null)
		{
			dateMax = new Date();
			dateMax.setTime(Long.MAX_VALUE);
		}
		double cash = 0;
		for(Cost cost : this.costs)
		{
			if(cost.getDate().getTime() >= dateMin.getTime() && cost.getDate().getTime() <= dateMax.getTime()) cash += cost.getPrice();
		}
		return cash;
	}
	
	/**
	 * Zwraca sume kosztow za licencje z przedzialu miedzy minimalna i maksymalna daty.
	 *
	 * @param dateMin minimalna data.
	 * @param dateMax maksymalna data.
	 * @return sume kosztow za licencje z zadanego przedzialu.
	 */
	public double getFilmCosts(Date dateMin, Date dateMax)
	{
		if(dateMin == null)
		{
			dateMin = new Date();
			dateMin.setTime(Long.MIN_VALUE);
		}
		
		if(dateMax == null)
		{
			dateMax = new Date();
			dateMax.setTime(Long.MAX_VALUE);
		}
		double cash = 0;
		for(Cost cost : this.costs)
		{
			if(cost.getDate().getTime() >= dateMin.getTime() && cost.getDate().getTime() <= dateMax.getTime() && cost.getType().equals("LICENCJA")) cash += cost.getPrice();
		}
		return cash;
	}
	
	/**
	 * Zwraca sume kosztow za seansy z przedzialu miedzy minimalna i maksymalna daty.
	 *
	 * @param dateMin minimalna data.
	 * @param dateMax maksymalna data.
	 * @return sume kosztow za seansy z zadanego przedzialu.
	 */
	public double getSeanceCosts(Date dateMin, Date dateMax)
	{
		if(dateMin == null)
		{
			dateMin = new Date();
			dateMin.setTime(Long.MIN_VALUE);
		}
		
		if(dateMax == null)
		{
			dateMax = new Date();
			dateMax.setTime(Long.MAX_VALUE);
		}
		double cash = 0;
		for(Cost cost : this.costs)
		{
			if(cost.getDate().getTime() >= dateMin.getTime() && cost.getDate().getTime() <= dateMax.getTime() && cost.getType().equals("SEANS")) cash += cost.getPrice();
		}
		return cash;
	}
	
	/**
	 * Zwraca sume kosztow za bilety z przedzialu miedzy minimalna i maksymalna daty.
	 *
	 * @param dateMin minimalna data.
	 * @param dateMax maksymalna data.
	 * @return sume kosztow za bilety z zadanego przedzialu.
	 */
	public double getTicketsCosts(Date dateMin, Date dateMax)
	{
		if(dateMin == null)
		{
			dateMin = new Date();
			dateMin.setTime(Long.MIN_VALUE);
		}
		
		if(dateMax == null)
		{
			dateMax = new Date();
			dateMax.setTime(Long.MAX_VALUE);
		}
		double cash = 0;
		for(Cost cost : this.costs)
		{
			if(cost.getDate().getTime() >= dateMin.getTime() && cost.getDate().getTime() <= dateMax.getTime() && cost.getType().equals("BILET")) cash += cost.getPrice();
		}
		return cash;
	}
	
	/* (non-Javadoc)
	 * @see java.lang.Object#toString()
	 */
	@Override
	public String toString()
	{
		String myString = "_________________________________________________ \n"
						+ "   TYP   |              DATA             | PRZYCH”D/STRATA \n"
						+ "_________________________________________________ \n";
		for(Cost cost : costs) myString = myString + cost.toString() + "\n";
		return myString;
	}
}
