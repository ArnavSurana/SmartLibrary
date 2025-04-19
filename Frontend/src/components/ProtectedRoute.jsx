import React,{useState,useEffect} from 'react'
import {useSelector} from 'react-redux'
import {useNavigate} from 'react-router-dom'
import CircularProgress from '@mui/material/CircularProgress';
import Box from '@mui/material/Box';
export default function ProtectedRoute({children,authentication=true}) {
  const navigate= useNavigate()
  const [loader,setLoader]=useState(true)
  const authStatus =useSelector((state)=>state.auth.status)

    useEffect(()=>{
        if(authentication && authStatus!==authentication){
            navigate('/login')
        }
        else if(!authentication && authStatus!==authentication){
           navigate('/')
        }
        setLoader(false)
    },[authStatus,navigate,authentication])
    if (loader) {
        return (
          <Box 
            display="flex" 
            justifyContent="center" 
            alignItems="center" 
            minHeight="100vh"
          >
            <CircularProgress />
          </Box>
        );
      }

    return <div>{children}</div>
}

