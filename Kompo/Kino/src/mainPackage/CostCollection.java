package mainPackage;

import java.util.ArrayList;

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
}
