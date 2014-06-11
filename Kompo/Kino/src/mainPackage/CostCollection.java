package mainPackage;

import java.util.ArrayList;

/**
 * 
 * Klasa przechowuje wydatki b�d� przychody, kt�re u�ytkownik kupi b�d� zarezerwuje, zale�nie od u�ycia
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
