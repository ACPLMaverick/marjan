package mainPackage.Model;

import java.util.Date;

/**
 * 
 * Reperezentuje pojedynczy film
 *
 */
public class Film {
	
	private String myGenre;
	private String title;
	private double price;
	private double licensePrice;
	
	public Film()
	{
		this.title = "brak";
		this.myGenre = "nieznany";
		this.price = 0.0;
		this.licensePrice = 0.0;
	}
	
	public Film(String title, String myGenre, double price, double licensePrice)
	{
		this.title = title;
		this.myGenre = myGenre;
		this.price = price;
		this.licensePrice = licensePrice;
	}
	
	public String getTitle() { return this.title; }
	public String getGenre() { return this.myGenre; }
	public double getPrice() { return this.price; }
	public double getLicensePrice() { return this.licensePrice; }
	
	public String getPriceAsString() { return String.valueOf(this.price); }
	public String getLicensePriceAsString() { return String.valueOf(this.licensePrice); }
}
