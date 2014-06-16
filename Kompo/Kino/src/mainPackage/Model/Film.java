package mainPackage.Model;

import java.util.Date;

// TODO: Auto-generated Javadoc
/**
 * Klasa reprezentuje pojedynczy film.
 */
public class Film {
	
	private String myGenre;
	private String title;
	private double price;
	private double licensePrice;
	
	/** Stala okreslajaca ilosc pol jakie opisuja film w interfejsie. */
	public static final int fieldsCount = 4;
	
	/**
	 * Tworzy nowy obiekt typu Film.
	 */
	public Film()
	{
		this.title = "brak";
		this.myGenre = "nieznany";
		this.price = 0.0;
		this.licensePrice = 0.0;
	}
	
	/**
	 * Tworzy nowy obiekt typu Film z konkretnymi parametrami.
	 *
	 * @param title tytul filmu.
	 * @param myGenre gatunek filmu.
	 * @param price cena biletu.
	 * @param licensePrice cena licencji.
	 */
	public Film(String title, String myGenre, double price, double licensePrice)
	{
		this.title = title;
		this.myGenre = myGenre;
		this.price = price;
		this.licensePrice = licensePrice;
	}
	
	/**
	 * Zwraca tytul filmu.
	 *
	 * @return Tytul filmu.
	 */
	public String getTitle() { return this.title; }
	
	/**
	 * Zwraca gatunek filmu.
	 *
	 * @return Gatunek filmu.
	 */
	public String getGenre() { return this.myGenre; }
	
	/**
	 * Zwraca cene biletu.
	 *
	 * @return Cene biletu.
	 */
	public double getPrice() { return this.price; }
	
	/**
	 * Zwraca cene licencji.
	 *
	 * @return Cene licencji.
	 */
	public double getLicensePrice() { return this.licensePrice; }
	
	/**
	 * Zwraca cene biletu jako String.
	 *
	 * @return Cene biletu jako String.
	 */
	public String getPriceAsString() { return String.valueOf(this.price); }
	
	/**
	 * Zwraca cene licencji jako String.
	 *
	 * @return Cene licencji jako String.
	 */
	public String getLicensePriceAsString() { return String.valueOf(this.licensePrice); }

}
