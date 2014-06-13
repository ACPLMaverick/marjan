package mainPackage.Model;

import java.sql.Time;
import java.util.ArrayList;
import java.util.Date;

/**
 * 
 * Klasa przechowuje wydatki bπdü przychody, ktÛre uøytkownik kupi bπdü zarezerwuje, zaleønie od uøycia
 *
 */
public class CostCollection {
	private ArrayList<Cost> costs;
	
	public CostCollection()
	{
		costs = new ArrayList<Cost>();
	}
	
	public void add(Cost cost)
	{
		costs.add(cost);
	}
	
	public void delete(int i)
	{
		costs.remove(i);
	}
	
	public Cost get(int i)
	{
		return costs.get(i);
	}
	
	public ArrayList<Cost> get()
	{
		return costs;
	}
	
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
