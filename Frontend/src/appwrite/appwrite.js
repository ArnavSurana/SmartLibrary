import conf from '../conf/conf.js';
import { Client, Account, Storage, Databases, ID, Query  } from "appwrite";


export class AuthService {
    client = new Client();
    account;
    storage;
    databases;

    constructor() {
        this.client
            .setEndpoint(conf.appwriteUrl)
            .setProject(conf.appwriteProjectId)
        this.account = new Account(this.client);
        this.storage = new Storage(this.client);
        this.databases = new Databases(this.client);
            
    }

    async createAccount({email, password, name}) {
        try {
            const userAccount = await this.account.create(ID.unique(), email, password, name);
            if (userAccount) {
                // call another method
                return this.login({email, password});
            } else {
               return  userAccount;
            }
        } catch (error) {
            throw error;
        }
    }

    async login({email, password}) {
        try {
            return await this.account.createEmailPasswordSession(email, password);
        } catch (error) {
            throw error;
        }
    }

    async getCurrentUser() {
        try {
            return await this.account.get();
        } catch (error) {
            console.log("Appwrite serive :: getCurrentUser :: error", error);
        }

        return null;
    }

    async logout() {

        try {
            await this.account.deleteSessions();
        } catch (error) {
            console.log("Appwrite serive :: logout :: error", error);
        }
    }

    async uploadPDF(file) {
        try {
          // First upload file to storage
          const fileResponse = await this.storage.createFile(
            conf.appwriteBucketId, // storage bucket ID
            ID.unique(),
            file
          );
          
          // Then create database entry
          const dbEntry = await this.databases.createDocument(
            conf.appwriteDatabaseId,
            conf.appwritepdfCollectionId,
            ID.unique(),
            {
              name: file.name,
              size: file.size,
              fileId: fileResponse.$id,
              userId: (await this.getCurrentUser()).$id,
              uploadDate: new Date().toISOString()
            }
          );
          
          return dbEntry;
        } catch (error) {
          console.error('Error uploading PDF:', error);
          throw error;
        }
      };
      
      async getUserPDFs (userId) {
        try {
          const response = await this.databases.listDocuments(
            conf.appwriteDatabaseId,
            conf.appwritepdfCollectionId,
            [Query.equal('userId', userId)]
          );
          return response.documents;
        } catch (error) {
          console.error('Error fetching user PDFs:', error);
          throw error;
        }
      };
      
      getPDFPreviewURL(fileId)  {
        
         return this.storage.getFileView(conf.appwriteBucketId, fileId);
        
      };
      
      getPDFViewURL(fileId){
       
          return this.storage.getFileView(conf.appwriteBucketId, fileId);
        
      };
      
      async deletePDF(fileId, documentId, userId, bookName) {
        try {
          // Call backend API to delete chunks from Pinecone
          await fetch('/books/delete', {
            method: 'POST',
            headers: {
              'Content-Type': 'application/x-www-form-urlencoded',
            },
            body: new URLSearchParams({
              user_id: userId,
              book_name: bookName,
            }),
          });

          // Delete from storage
          await this.storage.deleteFile(conf.appwriteBucketId, fileId);
          
          // Delete from database
          await this.databases.deleteDocument(
            conf.appwriteDatabaseId,
            conf.appwritepdfCollectionId,
            documentId
          );
          
          return true;
        } catch (error) {
          console.error('Error deleting PDF:', error);
          throw error;
        }
      };
      
      // QA history functions
      async saveQAPair(pdfId, question, answer){
        try {
          return await this.databases.createDocument(
            conf.appwriteDatabaseId,
            conf.appwriteqaCollectionId,
            ID.unique(),
            {
              pdfId: pdfId,
              question: question,
              answer: answer,
              userId: (await this.getCurrentUser()).$id,
              timestamp: new Date().toISOString()
            }
          );
        } catch (error) {
          console.error('Error saving QA pair:', error);
          throw error;
        }
      };
      
      async getPDFQuestions(pdfId, userId) {
        try {
          const response = await this.databases.listDocuments(
            conf.appwriteDatabaseId,
            conf.appwriteqaCollectionId,
            [
              Query.equal('pdfId', pdfId),
              Query.equal('userId', userId)
            ]
          );
          return response.documents;
        } catch (error) {
          console.error('Error fetching PDF questions:', error);
          throw error;
        }
      };
}

const authService = new AuthService();

export default authService
