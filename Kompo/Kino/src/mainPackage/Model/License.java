package mainPackage.Model;

public class License {
	
	private String filmName;
	private double price;
	
	public License(String filmName, double price)
	{
		this.filmName = filmName;
		this.price = price;
	}
	
	public String getFilmName() { return this.filmName; }
	public double getPrice() { return this.price; }
	public String getPriceAsString() { return String.format("%.2f", this.price) + " z³"; }
}
